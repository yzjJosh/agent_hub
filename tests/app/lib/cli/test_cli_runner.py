from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from tests.app.lib.conftest import mock_cli_proc

from app.lib.cli.cli_runner import run_cli


class TestRunCli:

    async def test_success_returns_trimmed_output(self) -> None:
        """Success returns stdout and stderr with outer whitespace stripped and exit code 0."""
        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=mock_cli_proc("  line1\nline2\n  ", " warn \n", 0))) as create:
            result = await run_cli(["tool", "--flag"], "label")
        assert result.stdout == "line1\nline2"
        assert result.stderr == "warn"
        assert result.returncode == 0
        assert create.call_args.args == ("tool", "--flag")
        assert create.call_args.kwargs == {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }

    async def test_empty_output(self) -> None:
        """Success with no output yields empty strings and exit code 0."""
        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=mock_cli_proc("   \n", "", 0))):
            result = await run_cli(["tool"])
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.returncode == 0

    async def test_nonzero_exit(self) -> None:
        """Non-zero exit keeps the raw outputs."""
        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=mock_cli_proc("out", "boom\n", 1))):
            result = await run_cli(["tool"], "label")
        assert result.stdout == "out"
        assert result.stderr == "boom"
        assert result.returncode == 1

    @pytest.mark.parametrize("timeout,expected_message", [
        (60, "label timed out after 60 seconds."),
        (5, "my search timed out after 5 seconds."),
    ])
    async def test_timeout(self, timeout: float, expected_message: str) -> None:
        """A hung command is killed and yields a non-zero result with a timeout message."""
        proc = mock_cli_proc()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            label = "label" if timeout == 60 else "my search"
            result = await run_cli(["tool"], label, timeout=timeout)
        assert result.stdout == ""
        assert result.stderr == expected_message
        assert result.returncode == -1
        proc.kill.assert_called_once()

    async def test_timeout_none_disables_timeout(self) -> None:
        """A slow command fails under a short timeout but succeeds with timeout=None."""
        async def _communicate() -> tuple[bytes, bytes]:
            await asyncio.sleep(0.15)
            return b"done\n", b""

        proc = mock_cli_proc()
        proc.communicate = _communicate
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            result = await run_cli(["tool"], "label", timeout=0.05)
        assert result.returncode == -1
        assert "timed out" in result.stderr

        proc2 = mock_cli_proc()
        proc2.communicate = _communicate
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc2)):
            result = await run_cli(["tool"], "label", timeout=None)
        assert result.stdout == "done"
        assert result.returncode == 0

    async def test_command_not_found(self) -> None:
        """A missing binary yields a non-zero result naming the command."""
        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(side_effect=FileNotFoundError("no such file: tool"))):
            result = await run_cli(["tool"], "label")
        assert result.stdout == ""
        assert result.stderr == "label could not be started: command not found"
        assert result.returncode == -1

    async def test_unexpected_error(self) -> None:
        """Any other spawn failure is reported in stderr with a non-zero code."""
        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(side_effect=OSError("spawn failed"))):
            result = await run_cli(["tool"], "label")
        assert result.stdout == ""
        assert result.stderr == "spawn failed"
        assert result.returncode == -1
