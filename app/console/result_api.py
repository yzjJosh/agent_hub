"""Endpoints for the result of a task run."""

from __future__ import annotations

import os

from starlette.responses import JSONResponse, Response

import app.lib.db.store as store_module
from app.console.errors import error_response
from app.lib.render import result_descriptor as rd
from app.lib.render import result_feed as rf
from app.lib.render.classify import RenderKind
from app.lib.render.detect import TEXT_FAMILY_KINDS
from app.lib.tool.base import parse_files_sidecar

__all__ = ["result_files_for", "result_feed_impl", "result_view_impl"]

_TEXT_KIND_TOKENS = {k.value for k in TEXT_FAMILY_KINDS}


async def result_files_for(data_dir: str, run_dir: str) -> list[str]:
    """Get the shared file paths saved with a run.

    Args:
        data_dir: Where run folders are kept.
        run_dir: The run's folder name.

    Returns:
        The paths that were saved, or an empty list when none were.
    """
    records = await parse_files_sidecar(os.path.join(data_dir, run_dir))
    return [record["path"] for record in records]


def _parse_view_params(params: dict) -> tuple[RenderKind | None, int, Response | None]:
    kind = None
    line_offset = 0
    if "line_offset" in params:
        try:
            line_offset = int(params.get("line_offset"))
        except (ValueError, TypeError):
            return None, 0, error_response(400, "bad_param", "line_offset must be a non-negative integer.")
        if line_offset < 0:
            return None, 0, error_response(400, "bad_param", "line_offset must be a non-negative integer.")
    kind_raw = params.get("kind")
    if kind_raw:
        if kind_raw not in _TEXT_KIND_TOKENS:
            return None, 0, error_response(
                400, "bad_param", f"kind must be one of the text-family kinds: {', '.join(sorted(_TEXT_KIND_TOKENS))}."
            )
        kind = RenderKind(kind_raw)
    return kind, line_offset, None


async def _view_context(task_id: str, params: dict) -> tuple[RenderKind | None, int, str,
                                                              RenderKind | None] | Response:
    """Check the view params and the task, and prepare what the view needs.

    Args:
        task_id: Which task to look at.
        params: The query values from the request.

    Returns:
        The kind, starting line, run folder, and the kind the tool declared,
        or an error reply.
    """
    kind, line_offset, err = _parse_view_params(params)
    if err is not None:
        return err
    view = await store_module.task_view(task_id)
    if view is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    declared = RenderKind(view.render_type) if view.render_type is not None else None
    return kind, line_offset, os.path.join(store_module.task_store.data_dir, view.run_dir), declared


async def _describe_result(task_id: str, context: tuple[RenderKind | None, int, str,
                                                         RenderKind | None]) -> Response:
    """Describe a run's result body.

    Args:
        task_id: Which task to look at.
        context: The kind, starting line, run folder, and the kind the tool declared.

    Returns:
        The description of the result, or an error reply.
    """
    kind, line_offset, run_dir, declared = context
    try:
        descriptor = await rd.build_result_descriptor(task_id, run_dir, line_offset=line_offset, kind=kind,
                                                      declared_kind=declared)
    except FileNotFoundError:
        return error_response(404, "result_not_found", "No result stored for this task.")
    return JSONResponse(descriptor)


async def _build_feed(task_id: str, context: tuple[RenderKind | None, int, str, RenderKind | None]) -> Response:
    """Describe a run's result feed, the body plus its files.

    Args:
        task_id: Which task to look at.
        context: The kind, starting line, run folder, and the kind the tool declared.

    Returns:
        The feed, or an error reply.
    """
    kind, line_offset, run_dir, declared = context
    try:
        feed = await rf.build_result_feed(task_id, run_dir, kind=kind, line_offset=line_offset,
                                          declared_kind=declared)
    except FileNotFoundError:
        return error_response(404, "result_not_found", "No result stored for this task.")
    return JSONResponse(feed)


async def result_view_impl(task_id: str, params: dict) -> Response:
    """Describe a task's result for the viewer.

    The request can ask for another text kind and for a starting line.

    Args:
        task_id: Which task to look at.
        params: The query values from the request.

    Returns:
        The description of the result, or an error reply.
    """
    context = await _view_context(task_id, params)
    if isinstance(context, Response):
        return context
    return await _describe_result(task_id, context)


async def result_feed_impl(task_id: str, params: dict) -> Response:
    """Describe a task's result feed, the body plus its files.

    Args:
        task_id: Which task to look at.
        params: The query values from the request.

    Returns:
        The feed, or an error reply.
    """
    context = await _view_context(task_id, params)
    if isinstance(context, Response):
        return context
    return await _build_feed(task_id, context)
