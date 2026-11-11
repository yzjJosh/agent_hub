from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.lib.agent.engine import AgentRun

__all__ = [
    "CollectorResult",
    "DATA_SOURCE_TOOLS",
    "EvidenceItem",
    "REF_RE",
    "ResearchResult",
    "ResearchTask",
    "Review",
    "CitedRun",
    "TaskState",
]

REF_RE = re.compile(r"\[\[ref:([^\]]+)\]\]")

DATA_SOURCE_TOOLS = frozenset(
    {"brave_search", "google_search", "x_search", "github", "web_fetch", "rednote_research"}
)


@dataclass
class ResearchTask:
    id: str
    description: str


@dataclass
class EvidenceItem:
    run_id: str
    tool_name: str


@dataclass
class TaskState:
    id: str
    description: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    summary: str = ""


@dataclass
class CollectorResult:
    agent_run: AgentRun
    summary: str = ""
    citation_ids: list[str] = field(default_factory=list)


@dataclass
class CitedRun:
    run_id: str
    tool_name: str
    params: dict
    run_dir: str


@dataclass
class Review:
    passed: bool
    reason: str
    action: Literal["rewrite", "collect", "approve"]
    feedback: str
    iteration: int


@dataclass
class ResearchResult:
    report: str
    status: Literal["complete", "max_iterations"]
    iterations: int
    review_reason: str
