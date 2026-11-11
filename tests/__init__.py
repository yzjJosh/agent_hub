"""Helpers shared by every test tree."""

import re
from dataclasses import dataclass
from pathlib import Path

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
_SERVER_PART = "Part 1"
_OTHER_PROGRAMS_PART = "Part 2"
_HOST_SCRIPTS_PART = "Part 3"
_PART_MARKER_RE = re.compile(r"^# === (Part \d+):")
_ENTRY_RE = re.compile(r"^# ([A-Z][A-Z0-9_]*)=$")


@dataclass(frozen=True)
class EnvExample:
    """The entries of the repository's .env.example.

    Attributes:
        names: Every entry name, in file order.
        server: The names of the server part.
        other_programs: The names of the part read by other programs.
        host_scripts: The names of the host scripts part.
    """

    names: list[str]
    server: set[str]
    other_programs: set[str]
    host_scripts: set[str]


def read_env_example() -> EnvExample:
    """The entries of the repository's .env.example, by part."""
    parts: dict[str, set[str]] = {}
    names = []
    part = ""
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        marker = _PART_MARKER_RE.match(line)
        if marker:
            part = marker.group(1)
        entry = _ENTRY_RE.match(line)
        if entry:
            names.append(entry.group(1))
            parts.setdefault(part, set()).add(entry.group(1))
    return EnvExample(
        names=names,
        server=parts.get(_SERVER_PART, set()),
        other_programs=parts.get(_OTHER_PROGRAMS_PART, set()),
        host_scripts=parts.get(_HOST_SCRIPTS_PART, set()),
    )
