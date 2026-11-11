from __future__ import annotations

import asyncio
from dataclasses import dataclass

__all__ = ["run_cli"]

_TIMEOUT = 60


@dataclass
class _CliResult:
    """What a command printed and the code it ended with."""

    stdout: str
    stderr: str
    returncode: int


async def run_cli(args: list[str], label: str = "command", timeout: float | None = _TIMEOUT) -> _CliResult:
    """Run a command and keep what it printed.

    Args:
        args: The command and its arguments.
        label: Name for the command, used in failure messages.
        timeout: Seconds before the command is killed. None waits forever.

    Returns:
        What the command printed and the code it ended with. Spare space at
        the ends is removed and nothing is cut short. If the command cannot
        start or runs too long, the code is -1 and stderr says why.

    Example:
        >>> result = await run_cli(["ffprobe", "-v", "error", "video.mp4"])
        >>> result.returncode
        0
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return _CliResult("", f"{label} could not be started: command not found", -1)
    except Exception as exc:
        return _CliResult("", str(exc), -1)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return _CliResult("", f"{label} timed out after {timeout} seconds.", -1)

    return _CliResult(
        stdout.decode("utf-8", "replace").strip(),
        stderr.decode("utf-8", "replace").strip(),
        proc.returncode,
    )
