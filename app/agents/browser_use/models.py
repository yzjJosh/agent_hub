"""Data types for the browser use agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Literal

__all__ = [
    "BrowserUseResult",
    "Snapshot",
    "TabSnapshot",
    "TransactionResult",
    "TransactionStatus",
]

_BrowserUseStatus = Literal["completed", "impossible", "incomplete"]


class TransactionStatus(Enum):
    """Why a transaction ended the way it did."""

    SUCCESS = auto()
    STALE = auto()
    CAP_REACHED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class TabSnapshot:
    """One tab's view in a snapshot."""

    target_id: str
    screenshot_path: str | None
    caption: str
    title: str = ""
    url: str = ""
    capture_started_at: datetime | None = None
    capture_finished_at: datetime | None = None
    screenshot_error: str = ""


@dataclass(frozen=True)
class Snapshot:
    """One transaction's view of the open tabs, keyed by target id in tab order."""

    tabs: dict[str, TabSnapshot]
    checkpoint_id: str | None = None
    observed_at: datetime | None = None
    archive_error: str = ""


@dataclass(frozen=True)
class TransactionResult:
    """The outcome of one transaction, for the harness to act on."""

    status: TransactionStatus
    message: str
    snapshot: Snapshot
    files: tuple[str, ...]


@dataclass(frozen=True)
class BrowserUseResult:
    """The outcome of a browser use run."""

    status: _BrowserUseStatus
    summary: str
    files: tuple[str, ...]
