import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agents.browser_use.checkpoint_store import CaptureInterval, CheckpointStore, DomCapture, TabCapture
from app.agents.browser_use.models import TransactionResult, TransactionStatus
from app.agents.browser_use.session import Action, BrowserSession
from app.lib.browser.browser_manager import BrowserManager
from tests.app.agents.browser_use.conftest import FakeManager

_DOWNLOADS_DIR = "/tmp/downloads"
_BUDGET = 100
_QUARTER_RATIO = 0.25
_TENTH_RATIO = 0.1
_FIFTH_RATIO = 0.2
_HALF_RATIO = 0.5
_THREE_TENTH_RATIO = 0.3
_WHITE = 255
_SMALL_CHANGE_RATIO = 0.125
_LARGE_CHANGE_RATIO = 0.5
_THREE_TABS = 3
_OVERLAP_TIMEOUT_S = 5
_CAPTURE_HOLD_S = 60


async def _action_ok(_manager: BrowserManager, _tab: str | None) -> bool:
    """An action that always succeeds."""
    return True


async def _action_fail(_manager: BrowserManager, _tab: str | None) -> bool:
    """An action that always fails."""
    return False


async def _action_boom(_manager: BrowserManager, _tab: str | None) -> bool:
    """An action that always raises."""
    raise RuntimeError("boom")


def _counting_action() -> tuple[list[int], Action]:
    """Give back an action that records each run, and the run list."""
    calls: list[int] = []

    async def action(_manager: BrowserManager, _tab: str | None) -> bool:
        calls.append(1)
        return True

    return calls, action


def _seeing_action() -> tuple[list[str | None], Action]:
    """Give back an action that records each resolved tab, and the tab list."""
    seen: list[str | None] = []

    async def action(_manager: BrowserManager, tab: str | None) -> bool:
        seen.append(tab)
        return True

    return seen, action


async def _open_session(downloads_dir: str) -> BrowserSession:
    """Open a session pointed at a downloads folder."""
    session = BrowserSession(downloads_dir)
    await session.open()
    return session


async def _transact_within(session: BrowserSession, seconds: float) -> TransactionResult:
    """Run one plain transaction, and fail the test when it takes too long."""
    try:
        async with asyncio.timeout(seconds):
            return await session.transact(_action_ok, None)
    except TimeoutError:
        pytest.fail(f"the transaction did not finish in {seconds}s, the tab shots did not run at the same time")


async def _open_session_with_tab(fake_manager: FakeManager, downloads_dir: str) -> tuple[BrowserSession, str]:
    """Open a session with one tab, and give back the session and the tab."""
    tab = fake_manager.open_tab()
    session = await _open_session(downloads_dir)
    return session, tab


def _open_tabs(fake_manager: FakeManager, count: int) -> None:
    """Open the given number of tabs in the fake browser."""
    for _ in range(count):
        fake_manager.open_tab()


async def _transact_with_colors(
    fake_manager: FakeManager,
    session: BrowserSession,
    tab: str,
    colors: tuple[int, ...],
    *,
    target_id: str | None = None,
) -> list[TransactionStatus]:
    """Run one transaction per capture color, and give back the statuses."""
    statuses: list[TransactionStatus] = []
    for color in colors:
        fake_manager.capture_colors[tab] = color
        statuses.append((await session.transact(_action_ok, target_id)).status)
    return statuses


async def _use_up_budget(session: BrowserSession) -> TransactionResult:
    """Run one transaction per budget slot, and give back the last one."""
    result = await session.transact(_action_ok, None)
    for _ in range(_BUDGET - 1):
        result = await session.transact(_action_ok, None)
    assert result.status is TransactionStatus.SUCCESS
    return result


async def _transact_admitted(
    session: BrowserSession,
    target_id: str | None,
    *,
    new_tab: bool = False,
) -> TransactionResult:
    """Run one transaction, and check the guard let the action run."""
    calls, action = _counting_action()
    result = await session.transact(action, target_id, new_tab=new_tab)
    assert result.status is TransactionStatus.SUCCESS
    assert calls == [1]
    return result


async def _transact_blocked(
    session: BrowserSession,
    target_id: str | None,
) -> TransactionResult:
    """Run one transaction, and check the guard held the action back."""
    calls, action = _counting_action()
    result = await session.transact(action, target_id)
    assert result.status is TransactionStatus.STALE
    assert calls == []
    return result


class TestOpenClose:
    async def test_open_connects_manager_with_downloads_dir(self, fake_manager: FakeManager) -> None:
        session = BrowserSession(_DOWNLOADS_DIR)
        await session.open()
        assert fake_manager.creates == 1
        assert fake_manager.downloads_dir == _DOWNLOADS_DIR

    async def test_double_open_raises(self, fake_manager: FakeManager) -> None:
        session = BrowserSession(_DOWNLOADS_DIR)
        await session.open()
        with pytest.raises(RuntimeError, match="session is already open"):
            await session.open()
        assert fake_manager.creates == 1

    async def test_close_releases_manager(self, fake_manager: FakeManager) -> None:
        session = BrowserSession(_DOWNLOADS_DIR)
        await session.open()
        await session.close()
        assert fake_manager.cleanup_calls == 1

    async def test_double_close_is_noop(self, fake_manager: FakeManager) -> None:
        session = BrowserSession(_DOWNLOADS_DIR)
        await session.open()
        await session.close()
        await session.close()
        assert fake_manager.cleanup_calls == 1

    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda s: s.transact(_action_ok, None), id="transact"),
            pytest.param(lambda s: s.transact(_action_ok, "tab1", monitor_download=True), id="transact_download"),
            pytest.param(lambda s: s.get_downloaded_files(), id="get_downloaded_files"),
        ],
    )
    async def test_method_on_closed_session_raises(
        self,
        invoke: Callable[[BrowserSession], Awaitable[TransactionResult | list[str]]],
    ) -> None:
        session = BrowserSession(_DOWNLOADS_DIR)
        with pytest.raises(RuntimeError, match="session is not open"):
            await invoke(session)


class TestActionForwarding:
    @pytest.mark.parametrize(
        ("build_action", "record", "expected_args"),
        [
            pytest.param(
                lambda f: lambda m, tab: m.click_at_relative("tab1", _QUARTER_RATIO, _HALF_RATIO),
                lambda f: f.last_click,
                ("tab1", _QUARTER_RATIO, _HALF_RATIO),
                id="click",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.type_text_at_relative("tab1", _TENTH_RATIO, _FIFTH_RATIO, "hello"),
                lambda f: f.last_type_text,
                ("tab1", _TENTH_RATIO, _FIFTH_RATIO, "hello"),
                id="type_text",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.replace_input_text_at_relative("tab1", _TENTH_RATIO, _FIFTH_RATIO, "helo", "hello"),
                lambda f: f.last_replace_input_text,
                ("tab1", _TENTH_RATIO, _FIFTH_RATIO, "helo", "hello"),
                id="replace_input_text",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.scroll_by_relative("tab1", 0.0, _HALF_RATIO),
                lambda f: f.last_scroll,
                ("tab1", 0.0, _HALF_RATIO),
                id="scroll",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.press_key("tab1", "Enter"),
                lambda f: f.last_press_key,
                ("tab1", "Enter"),
                id="press_key",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.go_back("tab1"),
                lambda f: f.last_go_back,
                "tab1",
                id="go_back",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.go_forward("tab1"),
                lambda f: f.last_go_forward,
                "tab1",
                id="go_forward",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.close_page("tab1"),
                lambda f: f.last_close_page,
                "tab1",
                id="close_page",
            ),
            pytest.param(
                lambda f: lambda m, tab: m.solve_captcha("tab1", 7),
                lambda f: f.last_solve_captcha,
                ("tab1", 7),
                id="solve_captcha",
            ),
        ],
    )
    async def test_action_forwards_to_manager(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
        build_action: Callable[[FakeManager], Action],
        record: Callable[[FakeManager], tuple[str | float | int, ...] | str | None],
        expected_args: tuple[str | float | int, ...] | str,
    ) -> None:
        fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        result = await session.transact(build_action(fake_manager), None)
        assert record(fake_manager) == expected_args
        assert result.status is TransactionStatus.SUCCESS


class TestTransact:
    async def test_success_reports_status_and_snapshot(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        fake_manager.open_tab(title="Shop", url="https://a.com")
        session = await _open_session(str(tmp_path))
        result = await session.transact(_action_ok, None)
        assert result.status is TransactionStatus.SUCCESS
        assert result.message == ""
        assert result.files == ()
        assert list(result.snapshot.tabs) == ["tab1"]
        assert result.snapshot.tabs["tab1"].screenshot_path == str(tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id) / "01.jpg")

    async def test_cap_reached_skips_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        await _use_up_budget(session)
        calls, action = _counting_action()
        result = await session.transact(action, None)
        assert result.status is TransactionStatus.CAP_REACHED
        assert result.message == "step budget is used up"
        assert calls == []

    async def test_cap_returns_last_snapshot_after_budget_used(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        last = await _use_up_budget(session)
        result = await session.transact(_action_ok, None)
        assert result.status is TransactionStatus.CAP_REACHED
        assert result.snapshot is last.snapshot

    async def test_failed_action_reports_failed(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        result = await session.transact(_action_fail, None)
        assert result.status is TransactionStatus.FAILED
        assert result.message

    async def test_action_on_missing_tab_reports_failed(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        result = await session.transact(lambda m, tab: m.click_at_relative(tab, _QUARTER_RATIO, _HALF_RATIO), "gone")
        assert result.status is TransactionStatus.FAILED
        assert fake_manager.last_click is None

    async def test_action_exception_reports_failed(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        result = await session.transact(_action_boom, None)
        assert result.status is TransactionStatus.FAILED
        assert "boom" in result.message
        again = await session.transact(_action_ok, None)
        assert again.status is TransactionStatus.SUCCESS

    async def test_stale_screen_blocks_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, None)
        assert first.status is TransactionStatus.SUCCESS
        fake_manager.capture_colors[tab] = _WHITE
        calls, action = _counting_action()
        second = await session.transact(action, None)
        assert second.status is TransactionStatus.STALE
        assert second.message == "screen changed since you last saw it, action not run"
        assert second.files == ()
        assert calls == []
        assert second.snapshot is not first.snapshot

    async def test_unchanged_screen_admits_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        second = await session.transact(_action_ok, None)
        assert first.status is TransactionStatus.SUCCESS
        assert second.status is TransactionStatus.SUCCESS

    async def test_consecutive_stale_blocks_force_the_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, None)
        statuses = [first.status]
        statuses.extend(
            await _transact_with_colors(fake_manager, session, tab, (_WHITE, 0, _WHITE, 0, _WHITE, 0, _WHITE, 0))
        )
        assert statuses == [
            TransactionStatus.SUCCESS,
            TransactionStatus.STALE,
            TransactionStatus.STALE,
            TransactionStatus.STALE,
            TransactionStatus.SUCCESS,
            TransactionStatus.STALE,
            TransactionStatus.STALE,
            TransactionStatus.STALE,
            TransactionStatus.SUCCESS,
        ]

    async def test_stale_capture_of_closed_tab_is_ignored(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab_a = fake_manager.open_tab()
        tab_b = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, tab_b)
        fake_manager.capture_colors[tab_b] = _WHITE
        fake_manager.pages.pop(tab_b)
        second = await _transact_admitted(session, tab_b)
        assert first.status is TransactionStatus.SUCCESS
        assert list(second.snapshot.tabs) == [tab_a]

    async def test_small_screen_change_admits_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        fake_manager.capture_patches[tab] = (_SMALL_CHANGE_RATIO, _WHITE)
        await _transact_admitted(session, None)
        assert first.status is TransactionStatus.SUCCESS

    async def test_large_screen_change_blocks_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        fake_manager.capture_patches[tab] = (_LARGE_CHANGE_RATIO, _WHITE)
        await _transact_blocked(session, None)
        assert first.status is TransactionStatus.SUCCESS

    async def test_change_just_above_the_threshold_admits_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        fake_manager.capture_patches[tab] = (_QUARTER_RATIO, _WHITE)
        await _transact_admitted(session, None)
        assert first.status is TransactionStatus.SUCCESS

    async def test_change_just_below_the_threshold_blocks_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        fake_manager.capture_patches[tab] = (_THREE_TENTH_RATIO, _WHITE)
        await _transact_blocked(session, None)
        assert first.status is TransactionStatus.SUCCESS

    async def test_change_in_another_tab_admits_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab_a = fake_manager.open_tab()
        tab_b = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, tab_a)
        fake_manager.capture_colors[tab_b] = _WHITE
        await _transact_admitted(session, tab_a)
        assert first.status is TransactionStatus.SUCCESS

    async def test_change_in_the_target_tab_blocks_action(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab_a = fake_manager.open_tab()
        fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, tab_a)
        fake_manager.capture_colors[tab_a] = _WHITE
        calls, action = _counting_action()
        second = await session.transact(action, tab_a)
        assert first.status is TransactionStatus.SUCCESS
        assert second.status is TransactionStatus.STALE
        assert calls == []

    async def test_new_tab_action_skips_the_guard(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        fake_manager.capture_colors[tab] = _WHITE
        await _transact_admitted(session, None, new_tab=True)
        assert first.status is TransactionStatus.SUCCESS

    async def test_named_new_tab_action_skips_the_guard(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, tab)
        fake_manager.capture_colors[tab] = _WHITE
        seen, action = _seeing_action()
        second = await session.transact(action, tab, new_tab=True)
        assert first.status is TransactionStatus.SUCCESS
        assert second.status is TransactionStatus.SUCCESS
        assert seen == [tab]

    async def test_named_new_tab_action_resets_the_block_count(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, tab)
        blocked = await _transact_with_colors(fake_manager, session, tab, (_WHITE, 0))
        fake_manager.capture_colors[tab] = _WHITE
        await _transact_admitted(session, tab, new_tab=True)
        after = await _transact_with_colors(fake_manager, session, tab, (0, _WHITE, 0))
        assert first.status is TransactionStatus.SUCCESS
        assert blocked == [TransactionStatus.STALE, TransactionStatus.STALE]
        assert after == [TransactionStatus.STALE, TransactionStatus.STALE, TransactionStatus.STALE]

    async def test_blank_target_with_two_tabs_skips_the_guard(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab_a = fake_manager.open_tab()
        tab_b = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, tab_a)
        fake_manager.capture_colors[tab_a] = _WHITE
        fake_manager.capture_colors[tab_b] = _WHITE
        second = await session.transact(_action_ok, None)
        assert first.status is TransactionStatus.SUCCESS
        assert second.status is TransactionStatus.FAILED
        assert second.message == "action failed: 2 tabs are open, give the target id of the one to act on"

    async def test_blank_target_with_two_tabs_resets_the_block_count(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        tab_a = fake_manager.open_tab()
        fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        first = await session.transact(_action_ok, tab_a)
        blocked = await _transact_with_colors(fake_manager, session, tab_a, (_WHITE, 0), target_id=tab_a)
        fake_manager.capture_colors[tab_a] = _WHITE
        failed = await session.transact(_action_ok, None)
        after = await _transact_with_colors(fake_manager, session, tab_a, (0, _WHITE, 0), target_id=tab_a)
        assert first.status is TransactionStatus.SUCCESS
        assert blocked == [TransactionStatus.STALE, TransactionStatus.STALE]
        assert failed.status is TransactionStatus.FAILED
        assert failed.message == "action failed: 2 tabs are open, give the target id of the one to act on"
        assert after == [TransactionStatus.STALE, TransactionStatus.STALE, TransactionStatus.STALE]

    async def test_target_missing_from_the_last_view_skips_the_guard(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab_a = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, tab_a)
        tab_b = fake_manager.open_tab()
        fake_manager.capture_colors[tab_a] = _WHITE
        await _transact_admitted(session, tab_b)
        assert first.status is TransactionStatus.SUCCESS

    async def test_missing_stored_screenshot_counts_as_unchanged(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        first = await session.transact(_action_ok, None)
        stored = first.snapshot.tabs[tab].screenshot_path
        assert stored == os.path.join(str(tmp_path), "checkpoints", str(first.snapshot.checkpoint_id), "01.jpg")
        os.remove(stored)
        fake_manager.capture_colors[tab] = _WHITE
        await _transact_admitted(session, None)

    async def test_concurrent_transacts_do_not_interleave(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        events: list[str] = []

        async def action(_manager: BrowserManager, _tab: str | None) -> bool:
            events.append("start")
            await asyncio.sleep(0)
            events.append("end")
            return True

        results = await asyncio.gather(session.transact(action, None), session.transact(action, None))
        assert events == ["start", "end", "start", "end"]
        assert all(r.status is TransactionStatus.SUCCESS for r in results)


class TestTabResolution:
    async def test_blank_target_resolves_to_the_sole_tab(self, fake_manager: FakeManager, tmp_path: Path) -> None:
        tab = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        seen, action = _seeing_action()
        result = await session.transact(action, None)
        assert result.status is TransactionStatus.SUCCESS
        assert seen == [tab]

    async def test_blank_target_with_no_tabs_resolves_to_none(self, fake_manager: FakeManager, tmp_path: Path) -> None:
        assert not fake_manager.tab_ids()
        session = await _open_session(str(tmp_path))
        seen, action = _seeing_action()
        result = await session.transact(action, None)
        assert result.status is TransactionStatus.SUCCESS
        assert seen == [None]

    async def test_blank_target_with_two_tabs_fails_before_the_action(
        self, fake_manager: FakeManager, tmp_path: Path
    ) -> None:
        fake_manager.open_tab()
        fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        calls, action = _counting_action()
        result = await session.transact(action, None)
        assert result.status is TransactionStatus.FAILED
        assert "2 tabs are open, give the target id of the one to act on" in result.message
        assert calls == []

    @pytest.mark.parametrize("open_tabs", [1, 2], ids=["one_tab", "two_tabs"])
    async def test_blank_target_with_new_tab_resolves_to_none(
        self, fake_manager: FakeManager, tmp_path: Path, open_tabs: int
    ) -> None:
        _open_tabs(fake_manager, open_tabs)
        session = await _open_session(str(tmp_path))
        seen, action = _seeing_action()
        result = await session.transact(action, None, new_tab=True)
        assert result.status is TransactionStatus.SUCCESS
        assert seen == [None]

    async def test_blank_target_keeps_its_tab_when_a_second_tab_opens_mid_transaction(
        self, fake_manager: FakeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        await _transact_admitted(session, None)
        read_tabs = fake_manager.tab_ids

        def tab_ids_then_open_a_tab() -> list[str]:
            open_tabs = read_tabs()
            if len(open_tabs) == 1:
                fake_manager.open_tab()
            return open_tabs

        monkeypatch.setattr(fake_manager, "tab_ids", tab_ids_then_open_a_tab)
        seen, action = _seeing_action()
        result = await session.transact(action, None)
        assert result.status is TransactionStatus.SUCCESS
        assert seen == [tab]

    async def test_named_target_resolves_to_that_tab(self, fake_manager: FakeManager, tmp_path: Path) -> None:
        tab = fake_manager.open_tab()
        session = await _open_session(str(tmp_path))
        seen, action = _seeing_action()
        result = await session.transact(action, tab)
        assert result.status is TransactionStatus.SUCCESS
        assert seen == [tab]


class TestDownload:
    async def test_download_records_reserved_file(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        fake_manager.download_queue.append({"downloaded": True})
        calls, action = _counting_action()
        result = await session.transact(action, tab, monitor_download=True)
        reserved = result.files[0]
        assert result.status is TransactionStatus.SUCCESS
        assert result.files == (reserved,)
        assert calls == [1]
        record = fake_manager.last_download
        if record is None:
            raise AssertionError("download wait was not recorded")
        assert record[0] == tab
        assert record[1] is None
        assert record[3] == reserved
        assert record[4] == 30
        assert os.path.isfile(reserved)
        assert reserved.startswith(str(tmp_path) + os.sep)
        assert await session.get_downloaded_files() == [reserved]

    async def test_download_without_file_cleans_up_reservation(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        fake_manager.download_queue.append({"downloaded": False})
        result = await session.transact(_action_ok, tab, monitor_download=True)
        assert result.status is TransactionStatus.FAILED
        assert result.files == ()
        assert await session.get_downloaded_files() == []

    async def test_downloads_accumulate_across_transactions(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        fake_manager.download_queue.extend([{"downloaded": True}, {"downloaded": True}])
        first = await session.transact(_action_ok, tab, monitor_download=True)
        second = await session.transact(_action_ok, tab, monitor_download=True)
        assert await session.get_downloaded_files() == [first.files[0], second.files[0]]
        assert first.files[0] != second.files[0]

    async def test_download_failure_removes_reservation(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        fake_manager.download_exception = RuntimeError("dl failed")
        with pytest.raises(RuntimeError, match="dl failed"):
            await session.transact(_action_ok, tab, monitor_download=True)
        assert not (tmp_path / "txn_0" / "download").exists()

    async def test_download_refuses_unknown_tab(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        with pytest.raises(KeyError, match="No page found for target_id: gone"):
            await session.transact(_action_ok, "gone", monitor_download=True)
        assert fake_manager.last_download is None
        assert not (tmp_path / "txn_0" / "download").exists()

    async def test_download_blank_target_watches_the_sole_tab(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        fake_manager.download_queue.append({"downloaded": True})
        result = await session.transact(_action_ok, "", monitor_download=True)
        assert result.status is TransactionStatus.SUCCESS
        record = fake_manager.last_download
        if record is None:
            raise AssertionError("download wait was not recorded")
        assert record[0] == tab

    async def test_download_blank_target_with_no_tabs_fails(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        result = await session.transact(_action_ok, "", monitor_download=True)
        assert result.status is TransactionStatus.FAILED
        assert result.message == "action failed: no tab is open, open one with open_url first"
        assert fake_manager.last_download is None

    async def test_download_blank_target_with_two_tabs_fails(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        fake_manager.open_tab()
        fake_manager.open_tab()
        result = await session.transact(_action_ok, "", monitor_download=True)
        assert result.status is TransactionStatus.FAILED
        assert "2 tabs are open, give the target id of the one to act on" in result.message
        assert fake_manager.last_download is None


class TestSnapshot:
    async def test_snapshot_every_tab(self, fake_manager: FakeManager, tmp_path: Path) -> None:
        session = await _open_session(str(tmp_path))
        tab_a = fake_manager.open_tab(title="Shop", url="https://a.com")
        tab_b = fake_manager.open_tab(title="Cart", url="https://a.com/cart")
        result = await session.transact(_action_ok, None)
        run_dir = tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id)
        assert list(result.snapshot.tabs) == [tab_a, tab_b]
        assert result.snapshot.tabs[tab_a].screenshot_path == str(run_dir / "01.jpg")
        assert (run_dir / "01.jpg").exists()
        assert result.snapshot.tabs[tab_b].screenshot_path == str(run_dir / "02.jpg")
        assert (run_dir / "02.jpg").exists()
        assert fake_manager.last_screenshot == (tab_b, str(run_dir / "02.jpg"), False, _CAPTURE_HOLD_S, True)
        assert result.snapshot.tabs[tab_a].caption == f'1. {tab_a} — "Shop" — https://a.com — image 1'
        assert result.snapshot.tabs[tab_b].caption == f'2. {tab_b} — "Cart" — https://a.com/cart — image 2'

    async def test_snapshot_holds_each_shot_for_sixty_seconds(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        """Each tab's shot waits up to sixty seconds on the manager."""
        session, _tab = await _open_session_with_tab(fake_manager, str(tmp_path))
        await session.transact(_action_ok, None)
        record = fake_manager.last_screenshot
        if record is None:
            raise AssertionError("the manager never got a shot")
        assert record[3] == _CAPTURE_HOLD_S

    async def test_snapshot_middle_tab_failure_keeps_ordinal(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab_a = fake_manager.open_tab(title="A", url="https://a.com")
        tab_b = fake_manager.open_tab(title="B", url="https://b.com")
        tab_c = fake_manager.open_tab(title="C", url="https://c.com")
        fake_manager.capture_outcomes[tab_b] = "none"
        result = await session.transact(_action_ok, None)
        run_dir = tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id)
        assert result.snapshot.tabs[tab_b].screenshot_path is None
        assert result.snapshot.tabs[tab_b].caption == f"2. {tab_b} — no screenshot"
        assert result.snapshot.tabs[tab_a].caption == f'1. {tab_a} — "A" — https://a.com — image 1'
        assert result.snapshot.tabs[tab_c].caption == f'3. {tab_c} — "C" — https://c.com — image 2'
        assert sorted(p.name for p in run_dir.iterdir()) == ["01.jpg", "03.jpg"]

    async def test_snapshot_timeout_marks_tab_failed(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab_a = fake_manager.open_tab(title="A", url="https://a.com")
        tab_b = fake_manager.open_tab(title="B", url="https://b.com")
        fake_manager.capture_outcomes[tab_b] = "timeout"
        result = await session.transact(_action_ok, None)
        run_dir = tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id)
        assert result.snapshot.tabs[tab_b].screenshot_path is None
        assert result.snapshot.tabs[tab_b].caption == f"2. {tab_b} — no screenshot"
        assert result.snapshot.tabs[tab_a].screenshot_path == str(run_dir / "01.jpg")

    async def test_snapshot_hostile_title_stays_one_line(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab = fake_manager.open_tab(title="Real Page\n4. tabX — no screenshot — image 9", url="https://x.com")
        result = await session.transact(_action_ok, None)
        caption = result.snapshot.tabs[tab].caption
        assert "\n" not in caption
        assert caption == f'1. {tab} — "Real Page 4. tabX — no screenshot — image 9" — https://x.com — image 1'

    async def test_snapshot_strips_quotes_from_title(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab = fake_manager.open_tab(title='A "quoted" title', url="https://x.com")
        result = await session.transact(_action_ok, None)
        assert result.snapshot.tabs[tab].caption == f'1. {tab} — "A quoted title" — https://x.com — image 1'

    async def test_snapshot_truncates_long_title_and_url(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab = fake_manager.open_tab(title="T" * 150, url="https://x.com/" + "a" * 500)
        result = await session.transact(_action_ok, None)
        expected_title = "T" * 119 + "…"
        expected_url = "https://x.com/" + "a" * 285 + "…"
        assert result.snapshot.tabs[tab].caption == f'1. {tab} — "{expected_title}" — {expected_url} — image 1'

    async def test_snapshot_shoots_every_tab_at_the_same_time(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab_a = fake_manager.open_tab(title="A", url="https://a.com")
        tab_b = fake_manager.open_tab(title="B", url="https://b.com")
        tab_c = fake_manager.open_tab(title="C", url="https://c.com")
        fake_manager.capture_barrier = asyncio.Barrier(_THREE_TABS)
        result = await _transact_within(session, _OVERLAP_TIMEOUT_S)
        run_dir = tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id)
        assert result.snapshot.tabs[tab_a].screenshot_path == str(run_dir / "01.jpg")
        assert result.snapshot.tabs[tab_b].screenshot_path == str(run_dir / "02.jpg")
        assert result.snapshot.tabs[tab_c].screenshot_path == str(run_dir / "03.jpg")
        assert result.snapshot.tabs[tab_a].caption == f'1. {tab_a} — "A" — https://a.com — image 1'
        assert result.snapshot.tabs[tab_b].caption == f'2. {tab_b} — "B" — https://b.com — image 2'
        assert result.snapshot.tabs[tab_c].caption == f'3. {tab_c} — "C" — https://c.com — image 3'

    async def test_snapshot_one_failed_shot_keeps_the_tabs_beside_it(
        self,
        fake_manager: FakeManager,
        tmp_path: Path,
    ) -> None:
        session = await _open_session(str(tmp_path))
        tab_a = fake_manager.open_tab(title="A", url="https://a.com")
        tab_b = fake_manager.open_tab(title="B", url="https://b.com")
        tab_c = fake_manager.open_tab(title="C", url="https://c.com")
        fake_manager.capture_outcomes[tab_b] = "timeout"
        fake_manager.capture_barrier = asyncio.Barrier(_THREE_TABS)
        result = await _transact_within(session, _OVERLAP_TIMEOUT_S)
        run_dir = tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id)
        assert result.snapshot.tabs[tab_b].screenshot_path is None
        assert result.snapshot.tabs[tab_b].caption == f"2. {tab_b} — no screenshot"
        assert result.snapshot.tabs[tab_a].screenshot_path == str(run_dir / "01.jpg")
        assert result.snapshot.tabs[tab_c].screenshot_path == str(run_dir / "03.jpg")
        assert result.snapshot.tabs[tab_a].caption == f'1. {tab_a} — "A" — https://a.com — image 1'
        assert result.snapshot.tabs[tab_c].caption == f'3. {tab_c} — "C" — https://c.com — image 2'
        assert sorted(p.name for p in run_dir.iterdir()) == ["01.jpg", "03.jpg"]

    async def test_snapshot_carries_no_generation(self, fake_manager: FakeManager, tmp_path: Path) -> None:
        session = await _open_session(str(tmp_path))
        fake_manager.open_tab(title="A", url="https://a.com")
        result = await session.transact(_action_ok, None)
        assert (tmp_path / "checkpoints" / str(result.snapshot.checkpoint_id) / "01.jpg").exists()
        assert "generation" not in vars(result.snapshot)

    async def test_snapshot_without_tabs(self, fake_manager: FakeManager, tmp_path: Path) -> None:
        session = await _open_session(str(tmp_path))
        result = await session.transact(_action_ok, None)
        assert result.snapshot.tabs == {}
        assert fake_manager.screenshot_calls == 0


async def test_undecodable_screens_count_as_unchanged(fake_manager: FakeManager, tmp_path: Path) -> None:
    session, tab = await _open_session_with_tab(fake_manager, str(tmp_path))
    fake_manager.capture_outcomes[tab] = "garbage"
    first = await session.transact(_action_ok, None)
    second = await session.transact(_action_ok, None)
    assert first.status is TransactionStatus.SUCCESS
    assert second.status is TransactionStatus.SUCCESS


class TestCheckpointAcquisition:
    @pytest.mark.parametrize("action", [_action_ok, _action_fail, _action_boom])
    async def test_observation_preserves_source_outcome(
        self, fake_manager: FakeManager, tmp_path: Path, action: Action,
    ) -> None:
        """Saved observations keep the action outcome and image times."""
        tab = fake_manager.open_tab(title="Original", url="https://example.test/view")
        store, session = await _open_archived_session(tmp_path)
        result = await session.transact(action, tab, source_action="click")
        page = await store.list()
        record = page.rows[0].checkpoint
        captured = page.rows[0].tab
        assert result.snapshot.checkpoint_id == record.checkpoint_id
        assert result.snapshot.observed_at == record.created_at
        assert record.source_action == "click"
        assert record.status is result.status
        assert captured.target_id == tab
        assert captured.title == "Original"
        assert captured.url == "https://example.test/view"
        assert captured.screenshot.interval.started_at <= captured.screenshot.interval.finished_at
        assert captured.dom.unavailable_reason == "Not captured"

    async def test_stale_observations_never_overwrite_earlier_images(
        self, fake_manager: FakeManager, tmp_path: Path,
    ) -> None:
        """Each returned stale screen keeps its original image bytes."""
        tab = fake_manager.open_tab()
        store, session = await _open_archived_session(tmp_path)
        first = await session.transact(_action_ok, tab)
        original = Path(first.snapshot.tabs[tab].screenshot_path).read_bytes()
        fake_manager.capture_colors[tab] = _WHITE
        stale_one = await session.transact(_action_ok, tab)
        fake_manager.capture_colors[tab] = 0
        stale_two = await session.transact(_action_ok, tab)
        fake_manager.capture_colors[tab] = _WHITE
        stale_three = await session.transact(_action_ok, tab)
        assert stale_one.status is TransactionStatus.STALE
        assert stale_two.status is TransactionStatus.STALE
        assert stale_three.status is TransactionStatus.STALE
        identities = {first.snapshot.checkpoint_id, stale_one.snapshot.checkpoint_id,
                      stale_two.snapshot.checkpoint_id, stale_three.snapshot.checkpoint_id}
        assert len(identities) == 4
        image = await store.read_screenshot(first.snapshot.checkpoint_id, tab)
        assert Path(image.path).read_bytes() == original
        assert Path(stale_one.snapshot.tabs[tab].screenshot_path).read_bytes() != original
        assert len((await store.list()).rows) == 4

    @pytest.mark.parametrize("text", ["", "## Page text\n\nExact words."])
    async def test_dom_is_published_with_its_snapshot(
        self, fake_manager: FakeManager, tmp_path: Path, text: str,
    ) -> None:
        """A captured empty page remains different from absent page text."""
        tab = fake_manager.open_tab()
        capture = _captured_dom(tab, text)
        store, session = await _open_archived_session(tmp_path)
        result = await session.transact(_action_ok, tab, source_action="dump_page", capture_dom=lambda: capture)
        saved = await store.read_dom(result.snapshot.checkpoint_id, tab)
        assert saved.text == text
        assert saved.total == len(text)
        assert saved.tab.dom.interval == capture.dom.interval
        assert saved.tab.dom.interval.finished_at <= saved.tab.screenshot.interval.started_at

    @pytest.mark.parametrize("leave_other_tab", [False, True])
    async def test_closed_dump_target_remains_archived(
        self, fake_manager: FakeManager, tmp_path: Path, leave_other_tab: bool,
    ) -> None:
        """A dump target can be absent from the final current tab set."""
        target = fake_manager.open_tab()
        remaining = fake_manager.open_tab() if leave_other_tab else None
        capture = _captured_dom(target, "Captured before closing")
        store, session = await _open_archived_session(tmp_path)

        async def close_target(manager: BrowserManager, tab: str | None) -> bool:
            return await manager.close_page(tab)

        result = await session.transact(close_target, target, source_action="dump_page", capture_dom=lambda: capture)
        assert list(result.snapshot.tabs) == ([remaining] if leave_other_tab else [])
        saved = await store.read_dom(result.snapshot.checkpoint_id, target)
        assert saved.text == "Captured before closing"
        assert saved.tab.screenshot.path is None
        assert saved.tab.title == "Captured title"

    async def test_partial_screenshot_failure_keeps_dom(
        self, fake_manager: FakeManager, tmp_path: Path,
    ) -> None:
        """Failed images do not remove successfully captured text."""
        tab = fake_manager.open_tab()
        fake_manager.capture_outcomes[tab] = "none"
        capture = _captured_dom(tab, "Still available")
        store, session = await _open_archived_session(tmp_path)
        result = await session.transact(_action_ok, tab, capture_dom=lambda: capture)
        saved = await store.read_dom(result.snapshot.checkpoint_id, tab)
        assert saved.text == "Still available"
        assert saved.tab.screenshot.unavailable_reason == "Screenshot capture failed"
        assert result.status is TransactionStatus.SUCCESS

    async def test_cap_response_reuses_checkpoint(
        self, fake_manager: FakeManager, tmp_path: Path,
    ) -> None:
        """The step limit returns the last actual observation."""
        fake_manager.open_tab()
        store, session = await _open_archived_session(tmp_path)
        last = await _use_up_budget(session)
        capped = await session.transact(_action_ok, None)
        assert capped.status is TransactionStatus.CAP_REACHED
        assert capped.snapshot == last.snapshot
        assert len((await store.list(limit=_BUDGET)).rows) == _BUDGET

    async def test_archive_failure_preserves_executed_action(
        self, fake_manager: FakeManager, tmp_path: Path,
    ) -> None:
        """Archive failure does not report that the browser action failed."""
        tab = fake_manager.open_tab()
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        store = CheckpointStore(str(blocked))
        session = BrowserSession(str(tmp_path / "downloads"), checkpoint_store=store)
        await session.open()
        calls, action = _counting_action()
        result = await session.transact(action, tab)
        assert calls == [1]
        assert result.status is TransactionStatus.SUCCESS
        assert result.snapshot.checkpoint_id is None
        assert "do not repeat an executed action" in result.snapshot.archive_error
        assert (await store.list()).rows == ()
        assert list(result.snapshot.tabs) == [tab]

    async def test_cancelled_capture_is_not_published(
        self, fake_manager: FakeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation leaves earlier evidence and removes unfinished captures."""
        tab = fake_manager.open_tab()
        store, session = await _open_archived_session(tmp_path)
        first = await session.transact(_action_ok, tab)
        started = asyncio.Event()

        async def hold_capture(
            _target_id: str, _path: str, **_options: bool | float,
        ) -> str | None:
            started.set()
            await asyncio.Event().wait()
            return None

        monkeypatch.setattr(fake_manager, "screenshot", hold_capture)
        pending = asyncio.create_task(session.transact(_action_ok, None, new_tab=True))
        await asyncio.wait_for(started.wait(), _OVERLAP_TIMEOUT_S)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert len((await store.list()).rows) == 1
        saved = await store.read_screenshot(first.snapshot.checkpoint_id, tab)
        assert Path(saved.path).exists()
        assert sorted(path.name for path in (tmp_path / "checkpoints").iterdir()) == [first.snapshot.checkpoint_id]


def _captured_dom(target_id: str, text: str) -> TabCapture:
    now = datetime.now(UTC)
    return TabCapture(target_id, "Captured title", "https://example.test", dom=DomCapture(text, CaptureInterval(now, now)))


async def _open_archived_session(directory: Path) -> tuple[CheckpointStore, BrowserSession]:
    store = CheckpointStore(str(directory))
    session = BrowserSession(str(directory), checkpoint_store=store)
    await session.open()
    return store, session
