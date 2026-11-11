from __future__ import annotations

import asyncio
import builtins
import contextvars
import io
import os
import queue
import sys
import threading
from typing import Callable, Protocol, TextIO, override

__all__ = [
    "LogBuffer",
    "active_log_buffer",
    "install_log_tee",
    "restore_log_tee",
]

active_log_buffer: contextvars.ContextVar[LogBuffer | None]
active_log_buffer = contextvars.ContextVar("active_log_buffer", default=None)

_original_print: Callable[..., None] | None = None


def _open_log(path: str | os.PathLike[str]) -> TextIO:
    """Open a log file for adding text at its end.

    Args:
        path: The file to open.

    Returns:
        The open file.
    """
    return open(os.fspath(path), "a", encoding="utf-8")


class LogBuffer:
    """A text buffer that also copies what it holds into a file.

    Use create to make one that is ready to take text. Writing to one never
    blocks.
    """

    def __init__(self) -> None:
        """Set up an empty buffer that is not taking text yet."""
        self._lock = threading.Lock()
        self._chunks: list[str] = []
        self._pending: queue.Queue[str | None] = queue.Queue()
        self._fh: TextIO | None = None
        self._writer = threading.Thread(target=self._drain, daemon=True)

    @classmethod
    async def create(cls, path: str | os.PathLike[str] | None = None) -> LogBuffer:
        """Make a buffer that is ready to take text.

        Args:
            path: The file to copy the text into, or None to keep the text in
                memory only.

        Returns:
            A buffer ready to use.

        Raises:
            OSError: If the file cannot be opened.
        """
        buffer = cls()
        await buffer._start(path)
        return buffer

    async def _start(self, path: str | os.PathLike[str] | None) -> None:
        """Open the file and start copying text into it.

        Args:
            path: The file to copy the text into, or None to keep the text in
                memory only.
        """
        if path is not None:
            self._fh = await asyncio.to_thread(_open_log, path)
        self._writer.start()

    def _drain(self) -> None:
        """Write waiting text to the file until the buffer is closed."""
        while True:
            text = self._pending.get()
            if text is None:
                break
            if self._fh is not None:
                self._fh.write(text)
                self._fh.flush()

    def write(self, text: str) -> None:
        if not text:
            return
        # The text kept in memory and the text put in the file stay in the same order.
        with self._lock:
            self._chunks.append(text)
            self._pending.put(text)

    async def read(self) -> str:
        """Get all the text the buffer holds.

        Returns:
            The text.
        """
        return await asyncio.to_thread(self._read)

    def _read(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    async def close(self) -> None:
        """Stop the buffer and close its file."""
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        self._pending.put(None)
        self._writer.join()
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class _Tee(io.TextIOBase):
    """A stream that writes to the real stream and to the active log buffer."""

    def __init__(self, original: io.TextIOBase) -> None:
        self.original = original

    @staticmethod
    def _unwrap(stream: io.TextIOBase) -> io.TextIOBase:
        """The stream inside a tee, or the given stream when it is not a tee."""
        if isinstance(stream, _Tee):
            return stream.original
        return stream

    @override
    def writable(self) -> bool:
        return True

    @override
    def write(self, s: str) -> int:
        buffer = active_log_buffer.get()
        if buffer is not None:
            buffer.write(s)
        return self.original.write(s)

    @override
    def flush(self) -> None:
        self.original.flush()


class _Printable(Protocol):
    """A value print writes to the stream."""

    def __str__(self) -> str:
        ...


def _tee_print(*args: _Printable, **kwargs: str | io.TextIOBase | bool) -> None:
    """Print that still reaches the active log buffer when stdout has been swapped out."""
    target = kwargs.get("file", sys.stdout)
    if isinstance(target, io.TextIOBase) and not isinstance(target, _Tee):
        if active_log_buffer.get() is not None:
            kwargs["file"] = _Tee(target)
    assert _original_print is not None
    _original_print(*args, **kwargs)


def install_log_tee() -> None:
    """Make stdout and stderr also write to the active log buffer."""
    global _original_print
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    if _original_print is None:
        _original_print = builtins.print
        builtins.print = _tee_print


def restore_log_tee() -> None:
    """Put stdout and stderr back the way they were."""
    global _original_print
    sys.stdout = _Tee._unwrap(sys.stdout)
    sys.stderr = _Tee._unwrap(sys.stderr)
    if _original_print is not None:
        builtins.print = _original_print
        _original_print = None
