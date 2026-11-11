"""Builds the composite a result feed shows, the body plus its files."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

from app.lib.render import result_descriptor as rd
from app.lib.render.classify import RenderKind
from app.lib.render.descriptor import build_descriptor, resolve_view
from app.lib.tool.base import parse_files_sidecar

__all__ = ["build_result_feed"]

_MTIME_FMT = "%b %d %H:%M"


def _fmt_local(mtime: int | None) -> str | None:
    if mtime is None:
        return None
    return datetime.fromtimestamp(mtime).strftime(_MTIME_FMT)


async def _file_entry(share_path: str, size: int | None, mtime: int | None) -> dict:
    entry = {
        "path": share_path, "name": os.path.basename(share_path), "present": False,
        "size": size, "mtime": mtime, "mtime_local": _fmt_local(mtime),
        "descriptor": None, "error": None,
    }
    full = await resolve_view(share_path)
    if full is None or not await asyncio.to_thread(os.path.isfile, full):
        return entry
    try:
        descriptor = await build_descriptor(share_path, full)
    except Exception:
        entry["present"] = True
        entry["error"] = "load_error"
        return entry
    entry["present"] = True
    entry["descriptor"] = descriptor
    entry["path"] = descriptor["path"]
    entry["size"] = descriptor["size"]          # live share facts win
    entry["mtime"] = descriptor["mtime"]
    entry["mtime_local"] = _fmt_local(descriptor["mtime"])
    return entry


async def build_result_feed(task_id: str, run_dir: str, *, kind: RenderKind | None = None,
                            declared_kind: RenderKind | None = None,
                            line_offset: int = 0, full: bool = False) -> dict:
    """Compose the body plus the files a run saved.

    Args:
        task_id: The run the result came from.
        run_dir: Where the run keeps its files.
        kind: Show the body as this kind, or None to work it out.
        declared_kind: The kind the tool fixed for its results, or None.
        line_offset: The line to start a code body at.
        full: True to show the whole body at once.

    Returns:
        The body descriptor plus one entry per file, in sidecar order.

    Raises:
        FileNotFoundError: If the run has no result file.
    """
    body = await rd.build_result_descriptor(task_id, run_dir, line_offset=line_offset,
                                            kind=kind, declared_kind=declared_kind, full=full)
    records = await parse_files_sidecar(run_dir)
    files = [await _file_entry(r["path"], r["size"], r["mtime"]) for r in records]
    return {"body": body, "files": files}
