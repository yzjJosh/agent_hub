"""Timestamps written out in local time."""

__all__ = [
    "LOCAL_TIME_FORMAT",
    "iso_to_local",
]

from datetime import datetime

LOCAL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def iso_to_local(iso: str | None) -> str | None:
    """Turn an ISO timestamp into local time text.

    Args:
        iso: The timestamp, or None.

    Returns:
        The local time text, or None when iso is None.
    """
    if iso is None:
        return None
    return datetime.fromisoformat(iso).astimezone().strftime(LOCAL_TIME_FORMAT)
