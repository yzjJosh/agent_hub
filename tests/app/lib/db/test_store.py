from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pytest

import app.lib.db.store as store_module


class _TaskStore(Protocol):
    """The task record store the fixture opens."""


class _ToolStore(Protocol):
    """The tool record store the fixture opens."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


async def _seed_completed(
    task: _TaskStore, tool: _ToolStore, now: datetime
) -> None:
    for tid, age in (("c0", 6), ("c1", 2), ("c2", 1)):
        await task.insert_running(tid, None, _iso(now - timedelta(hours=age)), f"runs/{tid}")
        await tool.insert(tid, "tool", "key", {})
        await task.finish(tid, "completed", ended_at=_iso(now))


@pytest.fixture
async def stores(tmp_path: Path) -> tuple[_TaskStore, _ToolStore]:
    """Open both stores on a throwaway database."""
    previous = store_module.task_store
    store_module.init_store(str(tmp_path / "records.db"))
    await store_module.task_store.open()
    yield store_module.task_store, store_module.tool_store
    await store_module.task_store.close()
    store_module.task_store = previous


class TestTaskRecordStore:
    async def test_insert_running_then_finish_completed(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        started = _iso(_now())
        await task.insert_running("t1", None, started, "runs/t1")
        record = await task.get("t1")
        assert record is not None and record.status == "running"
        assert record.parent_task_id is None
        assert record.started_at == started
        assert record.run_dir == "runs/t1"
        ended = _iso(_now())
        await task.finish("t1", "completed", ended_at=ended)
        record = await task.get("t1")
        assert record.status == "completed"
        assert record.ended_at == ended

    async def test_finish_failed_stores_error(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        await task.insert_running("t2", None, _iso(_now()), "runs/t2")
        await task.finish("t2", "failed", error="boom")
        record = await task.get("t2")
        assert record.error == "boom"

    async def test_get_missing_returns_none(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        assert await task.get("nope") is None

    async def test_list_orders_by_started_at(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        now = _now()
        await task.insert_running("b", None, _iso(now), "runs/b")
        await task.insert_running("a", None, _iso(now - timedelta(hours=1)), "runs/a")
        rows = await task.list()
        assert [r.task_id for r in rows] == ["a", "b"]

    async def test_list_subtasks_chronological_with_rowid_tiebreak(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        ts = _iso(_now())
        await task.insert_running("c1", "p", ts, "runs/c1")
        await task.insert_running("c2", "p", ts, "runs/c2")
        subs = await store_module.task_store.list_subtasks("p")
        assert [s.task_id for s in subs] == ["c1", "c2"]
        assert await store_module.task_store.list_subtasks("nobody") == []

    async def test_mark_orphans_failed(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        now = _iso(_now())
        await task.insert_running("r1", None, now, "runs/r1")
        await task.insert_running("r2", None, now, "runs/r2")
        await task.insert_running("c1", None, now, "runs/c1")
        await task.finish("c1", "completed", ended_at=now)
        marked = await task.mark_orphans_failed("server restarted before task completion")
        assert marked == 2
        assert (await task.get("r1")).status == "failed"
        assert (await task.get("r1")).error == "server restarted before task completion"
        assert (await task.get("c1")).status == "completed"

    async def test_no_orphans_returns_zero(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        assert await task.mark_orphans_failed("x") == 0

    async def test_delete_removes_row(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        started = _iso(_now())
        await task.insert_running("d1", None, started, "runs/d1")
        record = await task.get("d1")
        assert record is not None and record.status == "running"
        await task.delete("d1")
        assert await task.get("d1") is None

    async def test_delete_missing_is_noop(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        await task.delete("nope")
        assert await task.get("nope") is None

    async def test_data_dir(
            self, stores: tuple[_TaskStore, _ToolStore], tmp_path: Path) -> None:
        task, _ = stores
        assert task.data_dir == str(tmp_path)

    async def test_get_before_open_raises(self, tmp_path: Path) -> None:
        previous = store_module.task_store
        store_module.init_store(str(tmp_path / "records.db"))
        try:
            await store_module.task_store.close()
            with pytest.raises(RuntimeError, match="record store is not open"):
                await store_module.task_store.get("any")
        finally:
            store_module.task_store = previous


class TestToolRecordStore:
    async def test_insert_and_get_params_roundtrip(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _, tool = stores
        params = {"q": "中文", "n": 3, "flags": [1, 2]}
        await tool.insert("t3", "web_fetch", None, params)
        record = await tool.get("t3")
        assert record is not None and record.tool_name == "web_fetch"
        assert record.cache_key is None
        assert record.params == params

    async def test_get_missing_returns_none(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _, tool = stores
        assert await tool.get("nope") is None


class TestPresentationStore:
    """Tests for the presentation record store."""

    async def test_insert_and_get(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        recent = _iso(_now() - timedelta(days=1))
        await store_module.presentation_store.insert(
            "a" * 16, recent,
            "/view/presentation/aaaaaaaaaaaaaaaa", ["upload/a.png", "upload/b.md"],
        )
        p = await store_module.presentation_store.get("a" * 16)
        assert p is not None and p.id == "a" * 16
        assert p.presented_at == recent
        assert p.url_path == "/view/presentation/aaaaaaaaaaaaaaaa"
        assert p.files == ["upload/a.png", "upload/b.md"]
        assert p.task_id is None

    async def test_get_missing_returns_none(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        assert await store_module.presentation_store.get("b" * 16) is None

    async def test_insert_or_ignore_keeps_first(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        now = _now()
        await store_module.presentation_store.insert(
            "c" * 16, _iso(now - timedelta(days=2)), "/view/x", ["a.png"],
        )
        await store_module.presentation_store.insert(
            "c" * 16, _iso(now - timedelta(days=1)), "/view/y", ["b.png"],
        )
        p = await store_module.presentation_store.get("c" * 16)
        assert p.url_path == "/view/x"

    async def test_list_newest_orders_desc(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        now = _now()
        await store_module.presentation_store.insert(
            "a" * 16, _iso(now - timedelta(days=3)), "/view/a", ["a.png"],
        )
        await store_module.presentation_store.insert(
            "b" * 16, _iso(now - timedelta(days=2)), "/view/b", ["b.png"],
        )
        await store_module.presentation_store.insert(
            "c" * 16, _iso(now - timedelta(days=1)), "/view/c", ["c.png"],
        )
        results = await store_module.presentation_store.list_newest()
        assert [p.id for p in results] == ["c" * 16, "b" * 16, "a" * 16]

    async def test_newest_at(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        assert await store_module.presentation_store.newest_at() is None
        recent = _iso(_now() - timedelta(days=1))
        await store_module.presentation_store.insert(
            "a" * 16, recent, "/view/a", ["a"],
        )
        assert await store_module.presentation_store.newest_at() == recent

    async def test_delete_older(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        now = _now()
        old = _iso(now - timedelta(days=31))
        recent = _iso(now - timedelta(days=1))
        await store_module.presentation_store.insert("a" * 16, old, "/view/a", ["a"])
        await store_module.presentation_store.insert("b" * 16, recent, "/view/b", ["b"])
        removed = await store_module.presentation_store.delete_older(30)
        assert removed == 1
        assert await store_module.presentation_store.get("a" * 16) is None
        assert (await store_module.presentation_store.get("b" * 16)).presented_at == recent

    async def test_insert_with_task_id(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        await store_module.presentation_store.insert(
            "d" * 16, _iso(_now() - timedelta(days=1)), "/view/x", ["a"], task_id="task-1",
        )
        p = await store_module.presentation_store.get("d" * 16)
        assert p.task_id == "task-1"

    async def test_insert_with_title_roundtrips(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        await store_module.presentation_store.insert(
            "e" * 16, _iso(_now() - timedelta(days=1)), "/view/x", ["a"], title="Weekly report",
        )
        p = await store_module.presentation_store.get("e" * 16)
        assert p is not None and p.title == "Weekly report"

    async def test_insert_without_title_is_none(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _tool = stores
        await store_module.presentation_store.insert(
            "f" * 16, _iso(_now() - timedelta(days=1)), "/view/x", ["a"],
        )
        p = await store_module.presentation_store.get("f" * 16)
        assert p is not None and p.title is None


class TestToolStats:
    async def test_running_since_is_earliest_running_start(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        now = _now()
        await task.insert_running("r1", None, _iso(now - timedelta(hours=2)), "runs/r1")
        await tool.insert("r1", "tool", None, {})
        await task.insert_running("r2", None, _iso(now - timedelta(hours=1)), "runs/r2")
        await tool.insert("r2", "tool", None, {})
        await task.insert_running("c1", None, _iso(now - timedelta(hours=3)), "runs/c1")
        await tool.insert("c1", "tool", None, {})
        await task.finish("c1", "completed", ended_at=_iso(now))
        stats = await store_module.tool_stats()
        assert len(stats) == 1
        s = stats[0]
        assert (s["running"], s["completed"], s["failed"], s["cancelled"]) == (2, 1, 0, 0)
        assert s["running_since"] == _iso(now - timedelta(hours=2))
        assert s["last_activity"] == _iso(now - timedelta(hours=1))

    async def test_running_since_none_when_nothing_running(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        await _seed_completed(task, tool, _now())
        s = (await store_module.tool_stats())[0]
        assert s["running"] == 0
        assert s["running_since"] is None


class TestFindCache:
    async def test_latest_completed_within_window(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        now = _now()
        await _seed_completed(task, tool, now)
        hit = await tool.find_cache("tool", "key", _iso(now - timedelta(hours=24)))
        assert hit is not None and hit.task_id == "c2"
        assert hit.run_dir == "runs/c2"

    async def test_ignores_rows_before_window(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        now = _now()
        await task.insert_running("old", None, _iso(now - timedelta(hours=48)), "runs/old")
        await tool.insert("old", "tool", "key", {})
        await task.finish("old", "completed", ended_at=_iso(now))
        assert await tool.find_cache("tool", "key", _iso(now - timedelta(hours=24))) is None

    async def test_ignores_non_completed(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        now = _now()
        await task.insert_running("f", None, _iso(now), "runs/f")
        await tool.insert("f", "tool", "key", {})
        await task.finish("f", "failed", error="x")
        assert await tool.find_cache("tool", "key", _iso(now - timedelta(hours=24))) is None

    async def test_different_key_not_matched(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        now = _now()
        await task.insert_running("o", None, _iso(now), "runs/o")
        await tool.insert("o", "tool", "other", {})
        await task.finish("o", "completed", ended_at=_iso(now))
        assert await tool.find_cache("tool", "key", _iso(now)) is None


class TestAgentViewAxis:
    async def test_view_agent_axis_and_tri_way(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        await task.insert_running("parent", None, _iso(_now()), "runs/parent")
        await task.insert_running("agentchild", "parent", _iso(_now()), "runs/agentchild")
        await task.insert_running("toolchild", "parent", _iso(_now()), "runs/toolchild")
        await task.insert_running("plainchild", "parent", _iso(_now()), "runs/plainchild")
        await store_module.agent_store.insert("agentchild", "res")
        await tool.insert("toolchild", "web_search", None, {"q": "x"})

        av = await store_module.task_view("agentchild")
        assert av is not None and av.agent_name == "res"
        assert av.tool_name is None

        tv = await store_module.task_view("toolchild")
        assert tv is not None and tv.tool_name == "web_search"
        assert tv.agent_name is None

        pv = await store_module.task_view("plainchild")
        assert pv is not None and pv.tool_name is None and pv.agent_name is None

        subs = {v.task_id: v for v in await store_module.subtask_views("parent")}
        assert subs["agentchild"].agent_name == "res" and subs["agentchild"].tool_name is None
        assert subs["toolchild"].tool_name == "web_search" and subs["toolchild"].agent_name is None
        assert subs["plainchild"].tool_name is None and subs["plainchild"].agent_name is None


async def _seed_chain(task: _TaskStore, count: int, now: str) -> None:
    """A top-level task plus a parent-linked chain of `count - 1` children."""
    await task.insert_running("t0", None, now, "runs/t0")
    for i in range(1, count):
        await task.insert_running(f"t{i}", f"t{i - 1}", now, f"runs/t{i}")


class TestTaskLookup:
    """Prefix resolution and the ancestor walk over the shared view."""

    async def test_task_view_by_prefix_unique_match(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        full = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
        await task.insert_running(full, None, _iso(_now()), "runs/" + full)
        await store_module.agent_store.insert(full, "planner")
        view = await store_module.task_view_by_prefix("3f2504e0")
        assert view is not None and view.task_id == full
        assert view.agent_name == "planner"

    async def test_task_view_by_prefix_full_id(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        full = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
        await task.insert_running(full, None, _iso(_now()), "runs/" + full)
        view = await store_module.task_view_by_prefix(full)
        assert view is not None and view.task_id == full

    async def test_task_view_by_prefix_ambiguous_returns_none(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        now = _iso(_now())
        await task.insert_running("3f2504e0-1111-1111-1111-111111111111", None, now, "runs/a")
        await task.insert_running("3f2504e0-2222-2222-2222-222222222222", None, now, "runs/b")
        assert await store_module.task_view_by_prefix("3f2504e0") is None

    async def test_task_view_by_prefix_no_match_returns_none(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        await task.insert_running("3f2504e0-1111-1111-1111-111111111111", None, _iso(_now()), "runs/a")
        assert await store_module.task_view_by_prefix("00000000") is None

    async def test_ancestor_views_root_first_parent_last(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        now = _iso(_now())
        await task.insert_running("root", None, now, "runs/root")
        await task.insert_running("mid", "root", now, "runs/mid")
        await task.insert_running("leaf", "mid", now, "runs/leaf")
        await store_module.agent_store.insert("mid", "collector")
        chain = await store_module.ancestor_views("leaf")
        assert [v.task_id for v in chain] == ["root", "mid"]
        assert chain[1].agent_name == "collector" and chain[1].tool_name is None
        assert chain[0].status == "running"

    async def test_ancestor_views_top_level_is_empty(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        await task.insert_running("root", None, _iso(_now()), "runs/root")
        assert await store_module.ancestor_views("root") == []

    async def test_ancestor_views_unknown_task_is_empty(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        _task, _ = stores
        assert await store_module.ancestor_views("nobody") == []

    async def test_ancestor_views_cycle_does_not_loop(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        now = _iso(_now())
        await task.insert_running("a", "b", now, "runs/a")
        await task.insert_running("b", "a", now, "runs/b")
        chain = await store_module.ancestor_views("a")
        assert [v.task_id for v in chain] == ["b"]

    async def test_ancestor_views_depth_capped(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        now = _iso(_now())
        await _seed_chain(task, 70, now)
        chain = await store_module.ancestor_views("t69")
        assert len(chain) == 64
        assert chain[0].task_id == "t5"
        assert chain[-1].task_id == "t68"


class TestToolRecordRenderType:
    """Tests for storing and reading back the declared render kind."""

    async def test_set_render_type_then_task_view(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        await task.insert_running("rt1", None, _iso(_now()), "runs/rt1")
        await tool.insert("rt1", "tool", None, {})
        await tool.set_render_type("rt1", "markdown")
        view = await store_module.task_view("rt1")
        assert view is not None and view.render_type == "markdown"
        record = await tool.get("rt1")
        assert record is not None and record.task_id == "rt1"

    async def test_set_render_type_touches_only_its_row(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, tool = stores
        started = _iso(_now())
        await task.insert_running("rt1", None, started, "runs/rt1")
        await tool.insert("rt1", "tool", None, {})
        await task.insert_running("rt2", None, started, "runs/rt2")
        await tool.insert("rt2", "tool", None, {})
        await tool.set_render_type("rt1", "json")
        assert (await store_module.task_view("rt1")).render_type == "json"
        assert (await store_module.task_view("rt2")).render_type is None

    async def test_task_view_render_type_none_without_tool_row(
            self, stores: tuple[_TaskStore, _ToolStore]) -> None:
        task, _ = stores
        await task.insert_running("plain1", None, _iso(_now()), "runs/plain1")
        view = await store_module.task_view("plain1")
        assert view is not None and view.tool_name is None and view.render_type is None
