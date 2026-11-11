"""
Garbage Collector CLI

Cleans up each kind of leftover data.

  resources  files nobody has read or changed for N days, 30 by default
  runs       task and tool run records and run folders older than N days,
             30 by default
  render     saved renders whose source file is gone or changed, plus files
             nobody has read or changed for N days, 30 by default
   presentations  presentation rows older than N days, 30 by default

Run it with every category.

    python3 -u -m app.garbage_collector_cli

Options:
    --category all|resources|runs|render|presentations   category to sweep, all by default
    --days N                             staleness window for this run, each category has its own default

With one category it prints one removed path per line, or '(nothing to remove)'.
With every category it prints a '[<name>]' line, then that category's paths.
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

from app.lib.db.store import PRESENTATION_WINDOW_DAYS
from app.lib.gc.resources import sweep as sweep_resources
from app.lib.gc.presentations import sweep_presentations
from app.lib.gc.runs import sweep_runs
from app.lib.render.cache import sweep_sync as sweep_render
from app.lib.shared import paths

__all__ = []


@dataclass(frozen=True)
class _GcCategory:
    """One kind of leftover data the garbage collector cleans up."""

    name: str
    sweep: Callable[[int], list[str]]
    default_days: int


def _run_sweeps(categories: list[_GcCategory], days: int | None = None) -> dict[str, list[str]]:
    """Clean up every category given.

    A category that fails does not stop the others.

    Args:
        categories: The categories to clean up.
        days: How old something must be before it is removed. Each category
            uses its own default when this is None.

    Returns:
        The removed paths for each category name. A category that failed
        prints an error to stderr and maps to an empty list.
    """
    results: dict[str, list[str]] = {}
    for category in categories:
        try:
            results[category.name] = category.sweep(days if days is not None else category.default_days)
        except Exception as exc:
            print(f"[{category.name}] sweep failed: {exc}", file=sys.stderr)
            results[category.name] = []
    return results


def _categories() -> list[_GcCategory]:
    root = paths.resources_dir()
    data_dir = paths.data_dir()
    return [
        _GcCategory("resources", lambda d: sweep_resources(root, d), 30),
        _GcCategory("runs", lambda d: sweep_runs(d, data_dir), 30),
        _GcCategory("render", sweep_render, 30),
        _GcCategory("presentations", lambda d: sweep_presentations(d, data_dir), PRESENTATION_WINDOW_DAYS),
    ]


def main() -> None:
    """Read the command line and clean up the leftover data."""
    parser = argparse.ArgumentParser(prog="garbage_collector_cli")
    parser.add_argument("--category", choices=["all", "resources", "runs", "render", "presentations"], default="all")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    cats = _categories()
    if args.category != "all":
        cats = [c for c in cats if c.name == args.category]
    results = _run_sweeps(cats, args.days)

    if args.category == "all":
        for name, removed in results.items():
            print(f"[{name}]")
            for path in removed:
                print(path)
    else:
        removed = next(iter(results.values()))
        if removed:
            for path in removed:
                print(path)
        else:
            print("(nothing to remove)")


if __name__ == "__main__":
    main()
