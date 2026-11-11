from dataclasses import FrozenInstanceError
from typing import get_args, get_type_hints

import pytest

from app.agents.browser_use.models import (
    BrowserUseResult,
    Snapshot,
    TabSnapshot,
    TransactionResult,
    TransactionStatus,
)


class TestTabSnapshot:
    def test_fields(self) -> None:
        tab = TabSnapshot(target_id="tab1", screenshot_path="run/01.png", caption="1. tab1 — x")
        assert tab.target_id == "tab1"
        assert tab.screenshot_path == "run/01.png"
        assert tab.caption == "1. tab1 — x"

    def test_none_screenshot_path(self) -> None:
        tab = TabSnapshot(target_id="tab2", screenshot_path=None, caption="2. tab2 — no screenshot")
        assert tab.screenshot_path is None

    def test_frozen(self) -> None:
        tab = TabSnapshot(target_id="tab1", screenshot_path=None, caption="c")
        with pytest.raises(FrozenInstanceError):
            tab.target_id = "tab2"


class TestSnapshot:
    def test_fields_and_tab_order(self) -> None:
        snapshot = Snapshot(
            tabs={
                "a": TabSnapshot(target_id="a", screenshot_path="run/01.png", caption="1. a — x"),
                "b": TabSnapshot(target_id="b", screenshot_path=None, caption="2. b — no screenshot"),
            },
        )
        assert list(snapshot.tabs) == ["a", "b"]
        assert snapshot.tabs["b"].screenshot_path is None

    def test_frozen(self) -> None:
        snapshot = Snapshot(tabs={})
        with pytest.raises(FrozenInstanceError):
            snapshot.tabs = {"a": TabSnapshot(target_id="a", screenshot_path=None, caption="c")}


class TestTransactionStatus:
    def test_members(self) -> None:
        assert {s.name for s in TransactionStatus} == {"SUCCESS", "STALE", "CAP_REACHED", "FAILED"}


class TestTransactionResult:
    def test_fields(self) -> None:
        result = TransactionResult(
            status=TransactionStatus.SUCCESS,
            message="",
            snapshot=Snapshot(tabs={}),
            files=("txn_0/download",),
        )
        assert result.status is TransactionStatus.SUCCESS
        assert result.message == ""
        assert result.snapshot.tabs == {}
        assert result.files == ("txn_0/download",)

    def test_frozen(self) -> None:
        result = TransactionResult(TransactionStatus.FAILED, "no", Snapshot(tabs={}), ())
        with pytest.raises(FrozenInstanceError):
            result.status = TransactionStatus.SUCCESS


class TestBrowserUseStatus:
    def test_values(self) -> None:
        status = get_type_hints(BrowserUseResult)["status"]
        assert set(get_args(status)) == {"completed", "impossible", "incomplete"}


class TestBrowserUseResult:
    def test_fields(self) -> None:
        result = BrowserUseResult(status="completed", summary="done", files=("run/01.png", "run/02.png"))
        assert result.status == "completed"
        assert result.summary == "done"
        assert result.files == ("run/01.png", "run/02.png")

    def test_empty_files(self) -> None:
        result = BrowserUseResult(status="incomplete", summary="stopped", files=())
        assert result.files == ()

    def test_frozen(self) -> None:
        result = BrowserUseResult(status="impossible", summary="no way", files=())
        with pytest.raises(FrozenInstanceError):
            result.status = "completed"
