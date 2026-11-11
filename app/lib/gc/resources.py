"""Removes resources that nobody has opened or changed for a while."""

from app.lib.gc.file_sweep import sweep as sweep_files
from app.lib.shared.activity import last_used_sync

__all__ = ["sweep"]


def sweep(root: str, days: int) -> list[str]:
    """Remove resources with no recent interest or writes.

    Args:
        root: Resource folder to sweep.
        days: Days a resource may sit unused.

    Returns:
        Paths of removed files and empty folders.
    """
    return sweep_files(root, days, last_used=last_used_sync)
