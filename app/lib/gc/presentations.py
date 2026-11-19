"""Time-based cleanup of presentation rows."""

from __future__ import annotations

from app.lib.db.store import presentation_store
from app.lib.shared import paths

__all__ = ["sweep_presentations"]


def sweep_presentations(days: int, data_dir: str | None = None) -> list[str]:
    """Remove presentation rows older than the window.

    Args:
        days: A presentation older than this many days is removed.
        data_dir: Folder holding the records file.
            Defaults to the configured data folder.

    Returns:
        Ids of the removed presentation rows.
    """
    target_dir = paths.data_dir() if data_dir is None else data_dir
    return presentation_store.sweep_sync(days, target_dir)
