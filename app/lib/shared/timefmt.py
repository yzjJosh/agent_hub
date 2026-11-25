"""Timestamps written out in local time."""

__all__ = [
    "GeneratedTimestamps",
    "LOCAL_TIME_FORMAT",
    "build_generated_timestamps",
    "iso_to_local",
]

from dataclasses import dataclass
from datetime import datetime, timezone

LOCAL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class GeneratedTimestamps:
    """UTC and local time text for a response."""

    utc_iso: str
    local_display: str


def build_generated_timestamps() -> GeneratedTimestamps:
    """Build the current time in UTC and in local time.

    Returns:
        Both timestamp strings.
    """
    current_utc = datetime.now(timezone.utc)
    return GeneratedTimestamps(
        utc_iso=current_utc.isoformat(),
        local_display=current_utc.astimezone().strftime(LOCAL_TIME_FORMAT),
    )


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
