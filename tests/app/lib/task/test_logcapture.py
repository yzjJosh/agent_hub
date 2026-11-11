import asyncio
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest import CaptureFixture

from app.lib.task import logcapture


def _run_threads(func: Callable[[int], None], count: int) -> None:
    threads = [threading.Thread(target=func, args=(w,)) for w in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class TestLogBuffer:
    async def test_write_and_read(self) -> None:
        buf = await logcapture.LogBuffer.create()
        buf.write("hello ")
        buf.write("world")
        assert await buf.read() == "hello world"

    async def test_write_through_file(self, tmp_path: Path) -> None:
        path = tmp_path / "log"
        buf = await logcapture.LogBuffer.create(str(path))
        buf.write("line\n")
        await buf.close()
        assert path.read_text() == "line\n"

    async def test_write_empty_is_noop(self) -> None:
        buf = await logcapture.LogBuffer.create()
        buf.write("")
        assert await buf.read() == ""

    async def test_write_lands_in_file_before_close(self, tmp_path: Path) -> None:
        path = tmp_path / "log"
        buf = await logcapture.LogBuffer.create(str(path))
        try:
            buf.write("live\n")
            deadline = time.monotonic() + 5
            content = ""
            while time.monotonic() < deadline:
                content = path.read_text()
                if content == "live\n":
                    break
                time.sleep(0.01)
            assert content == "live\n"
        finally:
            await buf.close()

    async def test_concurrent_writers_all_flushed(self, tmp_path: Path) -> None:
        path = tmp_path / "log"
        buf = await logcapture.LogBuffer.create(str(path))

        def writer(worker: int) -> None:
            for i in range(25):
                buf.write(f"w{worker}-{i}\n")

        try:
            _run_threads(writer, 4)
        finally:
            await buf.close()
        lines = path.read_text().splitlines()
        assert len(lines) == 100
        assert (await buf.read()).splitlines() == lines


class TestTee:
    @pytest.fixture(autouse=True)
    def _tee(self) -> None:
        logcapture.install_log_tee()
        yield
        logcapture.restore_log_tee()

    async def test_routes_to_active_buffer(self, capfd: CaptureFixture[str]) -> None:
        buf = await logcapture.LogBuffer.create()
        token = logcapture.active_log_buffer.set(buf)
        try:
            print("to buffer")
        finally:
            logcapture.active_log_buffer.reset(token)
        assert "to buffer" in await buf.read()
        assert "to buffer" in capfd.readouterr().out

    def test_without_buffer_only_console(self, capfd: CaptureFixture[str]) -> None:
        assert logcapture.active_log_buffer.get() is None
        print("console only")
        out = capfd.readouterr().out
        assert "console only" in out

    async def test_stderr_routed_too(self, capfd: CaptureFixture[str]) -> None:
        buf = await logcapture.LogBuffer.create()
        token = logcapture.active_log_buffer.set(buf)
        try:
            print("to stderr", file=sys.stderr)
        finally:
            logcapture.active_log_buffer.reset(token)
        assert "to stderr" in await buf.read()
        assert "to stderr" in capfd.readouterr().err

    async def test_attribution_per_async_task(self) -> None:
        async def worker(buf: logcapture.LogBuffer, msg: str) -> None:
            token = logcapture.active_log_buffer.set(buf)
            try:
                print(msg)
            finally:
                logcapture.active_log_buffer.reset(token)

        a = await logcapture.LogBuffer.create()
        b = await logcapture.LogBuffer.create()
        await asyncio.gather(worker(a, "A"), worker(b, "B"))
        text_a, text_b = await a.read(), await b.read()
        assert "A" in text_a and "B" not in text_a
        assert "B" in text_b and "A" not in text_b

    async def test_to_thread_attribution(self) -> None:
        buf = await logcapture.LogBuffer.create()

        def sync_print() -> None:
            print("from thread")

        token = logcapture.active_log_buffer.set(buf)
        try:
            await asyncio.to_thread(sync_print)
        finally:
            logcapture.active_log_buffer.reset(token)
        assert "from thread" in await buf.read()

    def test_restore_is_idempotent_enough(self, capfd: CaptureFixture[str]) -> None:
        logcapture.restore_log_tee()
        print("after restore")
        assert "after restore" in capfd.readouterr().out
