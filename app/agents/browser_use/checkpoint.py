"""Read saved browser evidence from the current run."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import ClassVar, override

from app.agents.browser_use.checkpoint_store import (
    Artifact,
    Checkpoint,
    CheckpointError,
    CheckpointPage,
    CheckpointRow,
    CheckpointStore,
    CheckpointTab,
    DomWindow,
    ScreenshotRead,
)
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["CheckpointTool"]

_MAX_LIST_ROWS = 50
_MAX_DOM_CHAR = 25000
_MIN_DOM_CHAR = 500
_logger = logging.getLogger(__name__)


def _artifact_time(artifact: Artifact) -> str:
    if artifact.interval is None:
        return "Not captured"
    return f"{artifact.interval.started_at.isoformat()} to {artifact.interval.finished_at.isoformat()}"


def _artifact_summary(artifact: Artifact) -> str:
    if artifact.path is None:
        return f"Unavailable, {artifact.unavailable_reason}"
    length = f", {artifact.char_count} characters" if artifact.char_count is not None else ""
    return f"Captured{length}, {_artifact_time(artifact)}"


def _checkpoint_header(checkpoint: Checkpoint) -> str:
    return (
        f"Historical checkpoint: {checkpoint.checkpoint_id}\n"
        f"Observation time: {checkpoint.created_at.isoformat()}\n"
        f"Source action: {checkpoint.source_action}\n"
        f"Source outcome: {checkpoint.status.name.lower()}"
    )


def _tab_header(tab: CheckpointTab) -> str:
    return (
        f"Tab: {tab.target_id}\n"
        f"Title: {json.dumps(tab.title, ensure_ascii=False)}\n"
        f"URL: {json.dumps(tab.url, ensure_ascii=False)}"
    )


def _list_row(row: CheckpointRow) -> str:
    header = _checkpoint_header(row.checkpoint)
    tab = row.tab
    if tab is None:
        return f"{header}\nNo tabs or artifacts."
    return (
        f"{header}\n{_tab_header(tab)}\n"
        f"DOM: {_artifact_summary(tab.dom)}\n"
        f"DOM excerpt: {json.dumps(tab.dom.excerpt, ensure_ascii=False)}\n"
        f"Screenshot: {_artifact_summary(tab.screenshot)}"
    )


def _list_result(page: CheckpointPage) -> ToolResult:
    rows = "\n\n".join(_list_row(row) for row in page.rows) or "No checkpoints match this page."
    continuation = f"Next cursor: {page.next_cursor}." if page.next_cursor is not None else "Listing complete."
    return ToolResult(text=f"Saved browser evidence in this run.\n\n{rows}\n\n{continuation}")


def _dom_result(window: DomWindow) -> ToolResult:
    end = window.offset + len(window.text)
    continuation = f"Next offset: {window.next_offset}." if window.next_offset is not None else "DOM read complete."
    empty = "\nEmpty captured DOM." if window.total == 0 else ""
    header = (
        f"{_checkpoint_header(window.checkpoint)}\n{_tab_header(window.tab)}\n"
        "Historical evidence from an earlier observation, retained in context.\n"
        f"DOM capture interval: {_artifact_time(window.tab.dom)}\n"
        f"Total characters: {window.total}\nReturned range: [{window.offset}, {end}).{empty}"
    )
    return ToolResult(text=f"{header}\n\n{window.text}\n\nEnd historical DOM window. {continuation}")


def _screenshot_result(screenshot: ScreenshotRead) -> ToolResult:
    return ToolResult(
        text=(
            f"{_checkpoint_header(screenshot.checkpoint)}\n{_tab_header(screenshot.tab)}\n"
            "Historical evidence from an earlier observation, retained in context.\n"
            f"Screenshot capture interval: {_artifact_time(screenshot.tab.screenshot)}"
        ),
        files=(screenshot.path,),
    )


class CheckpointTool(Tool):
    """Read saved browser observations from one run."""

    name: ClassVar[str] = "checkpoint"
    tracked: ClassVar[bool] = True
    staleness: ClassVar[timedelta | None] = None
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Find and read browser evidence already captured in this run. "
        "Use before repeating navigation or dump_page to recover earlier evidence.\n\n"
        "list. Find saved observations.\n"
        "  required: action, string, list.\n"
        "  optional: target_id, string, empty for all tabs. cursor, integer, default 0. "
        "limit, integer, default 10, range 1 to 50.\n\n"
        "read. Read one saved artifact.\n"
        "  required: action, string, read. checkpoint_id, string. target_id, string.\n"
        "  optional: kind, string, dom or screenshot, default dom. "
        "For dom, offset, integer, default 0, and max_char, integer, default 25000, range 500 to 25000. "
        "Screenshot reads do not use DOM window parameters."
    )
    summary: ClassVar[str] = "Read saved browser evidence"
    params: ClassVar[tuple[Param, ...]] = (
        Param("action", str, enum=("list", "read"), description="Action to perform. Required by: list, read. Not used by: none."),
        Param("checkpoint_id", str, "", description="Saved observation to read. Required by: read. Not used by: list."),
        Param("target_id", str, "", description="Saved tab to read or filter. Required by: read. Used by: list. Not used by: none."),
        Param("kind", str, "dom", enum=("dom", "screenshot"), description="Artifact to read. Used by: read. Not used by: list."),
        Param("cursor", int, 0, min=0, description="Row offset after filtering tabs. Used by: list. Not used by: read."),
        Param("limit", int, 10, min=1, max=_MAX_LIST_ROWS, description="Maximum metadata rows. Used by: list. Not used by: read."),
        Param("offset", int, 0, min=0, description="First Unicode character to read. Used by: read of DOM. Not used by: list, screenshot reads."),
        Param("max_char", int, _MAX_DOM_CHAR, min=_MIN_DOM_CHAR, max=_MAX_DOM_CHAR, description="Maximum characters in a DOM window. Used by: read of DOM. Not used by: list, screenshot reads."),
    )

    def __init__(self, store: CheckpointStore) -> None:
        """Bind the tool to the current run.

        Args:
            store: Saved observations for this run.
        """
        self._store = store

    @override
    async def call(self, params: dict) -> ToolResult:
        """Find or read saved browser evidence.

        Args:
            params: The call arguments.

        Returns:
            Metadata or the requested historical evidence.

        Raises:
            ToolMessageError: If arguments or evidence are unavailable.
        """
        try:
            if params["action"] == "list":
                page = await self._store.list(params["target_id"], int(params["cursor"]), int(params["limit"]))
                return _list_result(page)
            if not params["checkpoint_id"].strip() or not params["target_id"].strip():
                raise ValueError("checkpoint_id and target_id are required for read")
            if params["kind"] == "screenshot":
                screenshot = await self._store.read_screenshot(params["checkpoint_id"], params["target_id"])
                return _screenshot_result(screenshot)
            window = await self._store.read_dom(
                params["checkpoint_id"], params["target_id"], int(params["offset"]), int(params["max_char"])
            )
            return _dom_result(window)
        except (CheckpointError, ValueError) as exc:
            raise ToolMessageError(f"Historical checkpoint request failed. {exc}") from exc
        except OSError as exc:
            _logger.exception("Cannot read saved browser evidence")
            raise ToolMessageError("Historical checkpoint evidence is unavailable") from exc
