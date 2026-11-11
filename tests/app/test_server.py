import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from asyncio import run as asyncio_run
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cloakbrowser import binary_info
from httpx import ASGITransport
from playwright.async_api import Page, Route as PlaywrightRoute, async_playwright, expect
from starlette.routing import Route
from starlette.testclient import TestClient
from websockets.asyncio.client import connect

import app.lib.db.store as store_module
import app.lib.render.thumbs as thumbs
import app.lib.task.logcapture as logcapture
from app.lib.browser.browser_manager import BrowserManager
from app.lib.shared.paths import download_dir
from app.server import build_app
from tests.app.conftest import (
    E2E_ENDED_AT,
    E2E_FAILED_250,
    E2E_PNG,
    E2E_RUNNING,
    E2E_STARTED_AT,
    e2e_page,
    log_lines,
    require_novnc,
    seed_task,
    skip_if_no_chrome,
    write_run_file,
    write_share_file,
)

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_REQUIRED_SETTINGS = (
    "REASON_BACKEND_URL", "REASON_THINKING_BACKEND_URL", "OCR_BACKEND_URL", "ASR_SERVER_URL",
    "EMBEDDING_SERVER_URL", "RERANKER_SERVER_URL", "COMFYUI_BACKEND_URL", "LOCAL_TTS_URL",
    "LOCAL_TTS_VOICE_EN", "LOCAL_TTS_VOICE_ZH", "BRAVE_SEARCH_API_KEY", "LAYA_SERVER_URL",
)


@pytest.fixture
def _no_required_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every setting that a tool needs."""
    for name in _REQUIRED_SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.timeout(120)
def test_app_builds_in_a_fresh_environment(_no_required_settings: None,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    """The app imports and builds with no settings set, and asks for none."""
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(_ROOT_DIR)}
    proc = subprocess.run(
        [sys.executable, "-c", "from app.server import build_app; build_app()"],
        env=env, cwd=_ROOT_DIR, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "is not set" not in proc.stdout
    assert "is not set" not in proc.stderr

    build_app()
    captured = capsys.readouterr()
    assert "is not set" not in captured.out
    assert "is not set" not in captured.err


@pytest.mark.timeout(60)
def test_lifespan_initializes_store_and_serves_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting the app opens the record store and serves the tool endpoint, and stopping closes the store."""
    monkeypatch.setattr(store_module, "_DB_PATH", str(tmp_path / "records.db"))

    with TestClient(build_app()) as client:
        assert store_module.task_store.data_dir == str(tmp_path)
        response = client.get("/mcp")
        # A plain GET does not ask for an event stream, so the answer is 406.
        assert response.status_code == 406
    with pytest.raises(RuntimeError, match="record store is not open"):
        asyncio_run(store_module.task_store.get("any"))


@pytest.mark.timeout(60)
def test_lifespan_restores_log_tee_when_startup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Printing is not captured after a failed startup."""
    (tmp_path / "records.db").mkdir()
    monkeypatch.setattr(store_module, "_DB_PATH", str(tmp_path / "records.db"))

    with pytest.raises(sqlite3.OperationalError):
        with TestClient(build_app()):
            pass
    buf = asyncio_run(logcapture.LogBuffer.create())
    token = logcapture.active_log_buffer.set(buf)
    try:
        print("no buffering")
    finally:
        logcapture.active_log_buffer.reset(token)
    assert asyncio_run(buf.read()) == ""


@pytest.mark.timeout(60)
def test_view_route_order_is_load_bearing() -> None:
    """The five view routes keep their order, with the catch all route last."""
    app = build_app()
    order = [r.path for r in app.routes if isinstance(r, Route) and r.path.startswith("/view")]
    assert order == [
        "/view/presentation/{id}",
        "/view/doc/{key}/{asset:path}",
        "/view/doc/{key}",
        "/view/result/{task_id}",
        "/view/{path:path}",
    ]


@pytest.mark.timeout(60)
def test_lifespan_starts_no_worker_and_stops_thumbnails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting the app starts nothing persistent and stopping the app stops the thumbnails."""
    monkeypatch.setattr(store_module, "_DB_PATH", str(tmp_path / "records.db"))
    stops = 0
    real_stop = thumbs.stop_thumbnails

    async def spy_stop() -> None:
        nonlocal stops
        await real_stop()
        stops += 1

    monkeypatch.setattr(thumbs, "stop_thumbnails", spy_stop)
    with TestClient(build_app()):
        assert stops == 0  # nothing persistent runs while the app is up
    assert stops == 1


# How the whole app is wired together.


async def test_spa_route_order_regression() -> None:
    app = build_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/tools"
        assert r.headers["cache-control"] == "no-cache"
        r = await c.get("/tools")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        r = await c.get("/tools/some/deep/link")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        r = await c.get("/resources")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        r = await c.get("/browser")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        r = await c.get("/assets/index.html")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        r = await c.get("/api/definitely-not-real")
        assert r.status_code == 404 and r.json()["code"] == "not_found"
        r = await c.get("/api")
        assert r.status_code == 404
        assert r.json() == {"error": "No API endpoint at '/api'.", "code": "not_found"}
        assert r.headers["x-frame-options"] == "DENY"
        assert "cache-control" not in r.headers
        r = await c.get("/assets/definitely-missing.css")
        assert r.status_code == 404 and "Agent Hub (placeholder" not in r.text
        r = await c.get("/vnc/vnc.html")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "max-age=3600"
        assert "x-frame-options" not in r.headers
    paths = [rt.path for rt in app.routes]
    assert {"/api", "/assets", "/vnc", "/dav", "/v1"} <= set(paths)


async def test_wrong_method_under_api_returns_json_envelope() -> None:
    """Any method on an unknown /api path gets a JSON 404 with no allowed list.

    A wrong method somewhere else still gets the plain text 405.
    """
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == {"error": "No API endpoint at '/api'.", "code": "not_found"}
        assert "allow" not in r.headers
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["x-content-type-options"] == "nosniff"
        r = await c.options("/api/foo")
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"
        assert "allow" not in r.headers
        r = await c.post("/api/foo")
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"
        r = await c.delete("/vnc/vnc.html")
        assert r.status_code == 405
        assert r.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize("newline", ["%0a", "%0d"])
async def test_encoded_newline_in_spa_path_serves_index(newline: str) -> None:
    """A path with an escaped newline serves the same page as a clean unknown path."""
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(f"/tools/web_fetch/aa{newline}bb")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["cache-control"] == "no-cache"
        clean = await c.get("/tools/some/deep/link")
        assert clean.status_code == 200
        assert r.text == clean.text


async def test_websockify_http_returns_404() -> None:
    """A plain request to /websockify gets a 404, not the app page."""
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/websockify")
        assert r.status_code == 404
        assert r.text == "Not Found"


async def test_vnc_csp_allows_data_images() -> None:
    """The screen sharing page may show images built into the page, every other page may not."""
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r_vnc = await c.get("/vnc/vnc.html")
        assert r_vnc.status_code == 200
        vnc_csp = r_vnc.headers["content-security-policy"]
        assert "img-src 'self' data:" in vnc_csp
        assert "default-src 'self'" in vnc_csp

        r_spa = await c.get("/vncfoo")
        assert r_spa.status_code == 200
        spa_csp = r_spa.headers["content-security-policy"]
        assert "img-src 'self'" in spa_csp
        assert "data:" not in spa_csp
        assert r_spa.headers["x-frame-options"] == "DENY"
        assert r_spa.headers["cache-control"] == "no-cache"

        r_api = await c.get("/api/definitely-not-real")
        api_csp = r_api.headers["content-security-policy"]
        assert "img-src 'self'" in api_csp
        assert "data:" not in api_csp


async def test_unmatched_path_serves_index_with_security_headers() -> None:
    """An unknown path serves the same page as /tools, with framing and caching off.

    The screen sharing page can still sit in a frame and is cached.
    """
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r_spa = await c.get("/no/such/spa/route")
        r_tools = await c.get("/tools")
        assert r_spa.status_code == 200
        assert r_spa.text == r_tools.text
        assert r_spa.headers["x-frame-options"] == "DENY"
        assert r_spa.headers["cache-control"] == "no-cache"
        assert r_tools.headers["x-frame-options"] == "DENY"
        assert r_tools.headers["cache-control"] == "no-cache"
        r_select = await c.get("/resources?select=downloads")
        assert r_select.status_code == 200
        assert r_select.text == r_tools.text
        r_api = await c.get("/api/definitely-not-real")
        assert r_api.status_code == 404
        assert r_api.headers["x-frame-options"] == "DENY"
        assert "cache-control" not in r_api.headers
        r_vnc = await c.get("/vnc/vnc.html")
        assert r_vnc.status_code == 200
        assert "x-frame-options" not in r_vnc.headers
        assert r_vnc.headers["cache-control"] == "max-age=3600"


@pytest.mark.parametrize("path", ["/mcpfoo", "/davfoo", "/v1foo"])
async def test_service_mount_name_prefix_catch_all_gets_security_headers(path: str) -> None:
    """A path that only starts like a service name serves the app page with framing and caching off."""
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        tools = await c.get("/tools")
        r = await c.get(path)
        assert r.status_code == 200
        assert r.text == tools.text
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["cache-control"] == "no-cache"
        assert "default-src 'self'" in r.headers["content-security-policy"]


@pytest.mark.parametrize(
    ("path", "status"),
    [("/dav", 200), ("/dav/upload", 301), ("/dav/x.bin", 404), ("/v1/audio/speech", 405)],
)
async def test_service_mounts_keep_skip_behavior(path: str, status: int) -> None:
    """The /dav and /v1 paths reach their own services and get none of the app page headers."""
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(path)
        assert r.status_code == status
        assert "x-frame-options" not in r.headers
        assert "cache-control" not in r.headers


async def test_dav_download_route_wins_over_dav_mount(dav_share: Path) -> None:
    """The /dav/dl download route wins even when a share file is named dl."""
    (dav_share / "dl").write_text("i am the share file named dl")
    (dav_share / "upload").mkdir()
    (dav_share / "upload" / "x.txt").write_text("hello download")
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/dav/dl", params={"path": "upload/x.txt"})
    assert r.status_code == 200
    assert r.text == "hello download"
    assert r.headers["content-disposition"] == 'attachment; filename="x.txt"'


# Checks on the console over the whole app.


_SHELL_ASSETS = (
    "/assets/css/tokens.css",
    "/assets/css/app.css",
    "/assets/js/main.js",
    "/assets/js/theme.js",
    "/assets/js/toast.js",
)
_CORE_ASSETS = (
    "/assets/js/format.js",
    "/assets/js/api.js",
    "/assets/js/ui.js",
    "/assets/js/state.js",
    "/assets/js/router.js",
)
_SCREEN_ASSETS = (
    "/assets/js/screens/tools.js",
    "/assets/js/screens/task-detail.js",
    "/assets/js/screens/resources.js",
    "/assets/js/screens/tool-tasks.js",
    "/assets/js/screens/params.js",
)
_MERGED_ASSETS = _SHELL_ASSETS + _CORE_ASSETS + _SCREEN_ASSETS


async def _assert_merged_assets(c: httpx.AsyncClient) -> None:
    """Check every console asset answers 200."""
    for asset in _MERGED_ASSETS:
        assert (await c.get(asset)).status_code == 200


async def test_shell_is_served() -> None:
    """The page at /tools is the exact index file and every asset it needs answers 200."""
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/tools")
        assert r.status_code == 200
        assert r.text == (await c.get("/assets/index.html")).text
        html = r.text
        assert "data-theme" in html and "agent-console-theme" in html
        assert 'id="sb-toggle"' in html and 'id="theme-toggle"' in html
        assert "/assets/css/tokens.css" in html and "/assets/js/main.js" in html
        assert "/assets/js/screens/tools.js" in html and "/assets/js/screens/task-detail.js" in html
        await _assert_merged_assets(c)
        assert (await c.get("/assets/css/tokens.css")).text.count("--bg-app") == 2


@pytest.mark.timeout(30)
async def test_routing_over_real_socket(app_http: SimpleNamespace) -> None:
    """Over a real port, / sends you to /tools, an unknown /api path gets a 404, and health answers 200."""
    async with httpx.AsyncClient(base_url=app_http.base_url, timeout=10) as c:
        r = await c.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/tools"
        r = await c.get("/api/definitely-not-real")
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"
        assert (await c.get("/api/health")).status_code == 200


@pytest.mark.timeout(30)
async def test_dav_serves_seeded_bytes(app_http: SimpleNamespace) -> None:
    """The file server hands back the exact bytes of a seeded file, and a missing path gets a 404."""
    rel = "downloads/e2e-seeded.png"
    write_share_file(app_http.share, rel, E2E_PNG)
    async with httpx.AsyncClient(base_url=app_http.base_url, timeout=10) as c:
        d = await c.get("/dav/" + rel)
        assert d.status_code == 200
        assert d.content[:4] == b"\x89PNG"
        assert (await c.get("/dav/nope/nope.bin")).status_code == 404


# --- Browser tests over the running app ---
#
# The browser is shared and stays running. Each test opens its own fresh window with
# nothing shared, leaves the shared window alone, and closes only what it opened.


_WAIT_TIMEOUT_MS = 10000
_MOBILE_VIEWPORT = {"width": 390, "height": 844}
_DESKTOP_VIEWPORT = {"width": 1024, "height": 768}
_WIDE_DESKTOP_VIEWPORT = {"width": 1440, "height": 900}
_POLL_SLEEP_S = 0.2
_MS_PER_S = 1000
Json = str | int | float | bool | None | list | Mapping


def _js_literal(value: Json) -> str:
    """Render a value as a JSON literal, which is also a JavaScript literal, to inline into an expression."""
    return json.dumps(value)


def _js_call(fn_js: str, arg: Json) -> str:
    """Build a zero-arg expression that runs an (arg) => expression with one inlined value."""
    return f"({fn_js}).call(null, {_js_literal(arg)})"


async def _wait_for(mgr: BrowserManager, tid: str, expression: str, timeout_ms: int) -> None:
    """Poll a tab until a page expression is truthy, and raise when the deadline passes.

    Args:
        mgr: The browser manager.
        tid: The tab to poll.
        expression: A zero-arg expression that is truthy when the condition holds.
        timeout_ms: How long to wait, in milliseconds.
    """
    deadline = time.monotonic() + timeout_ms / _MS_PER_S
    while True:
        if await mgr.evaluate(tid, expression):
            return
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for: " + expression[:120])
        await asyncio.sleep(_POLL_SLEEP_S)


async def _element_box(mgr: BrowserManager, tid: str, selector: str) -> dict:
    """The bounding box of the first element a selector points at.

    Args:
        mgr: The browser manager.
        tid: The tab to measure in.
        selector: Points at the element to measure.
    """
    return await mgr.evaluate(
        tid,
        "() => { const el = document.querySelector(" + _js_literal(selector) + ");"
        " const r = el.getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height}; }",
    )


async def _wait_new_page(mgr: BrowserManager, before: list[str], base_url: str, timeout_ms: int) -> str:
    """Wait until a link that opens a new tab does, and give back the URL it loaded.

    Only a tab that loads the stub's own origin counts as the new tab, so tabs
    that other tests open in the shared browser are not picked up. The tab is
    closed before the URL is given back.

    Args:
        mgr: The browser manager.
        before: The tab target ids that were open before the click.
        base_url: The origin a new tab must load before it is ours.
        timeout_ms: How long to wait, in milliseconds.
    """
    own = [tid for tid in before if (mgr.url(tid) or "").startswith(base_url)]
    deadline = time.monotonic() + timeout_ms / _MS_PER_S
    while True:
        fresh = [tid for tid in mgr.tab_ids() if tid not in own and (mgr.url(tid) or "").startswith(base_url)]
        if fresh:
            tid = fresh[0]
            try:
                await mgr.wait_for_load(tid, "domcontentloaded", timeout=timeout_ms)
                return mgr.url(tid)
            finally:
                await mgr.close_page(tid)
        if time.monotonic() >= deadline:
            raise AssertionError("no new tab opened")
        await asyncio.sleep(_POLL_SLEEP_S)


# The real mouse pointer is shared by every browser tab, so a hover can be lost
# when a tab opens or closes. A test that needs the hover to hold parks the
# pointer first and unparks it at the end.
_POINTER_EVENTS = ["mousemove", "mouseover", "mouseout", "mouseenter", "mouseleave"]
_POINTER_AWAY = (5, 5)
_POINTER_FREEZE_JS = """
(types) => {
  const swallow = (e) => e.stopImmediatePropagation();
  for (const t of types) window.addEventListener(t, swallow, true);
  window.__thawPointer = () => { for (const t of types) window.removeEventListener(t, swallow, true); };
}
"""


async def _park_pointer(mgr: BrowserManager, tid: str, selector: str) -> tuple[float, float]:
    """Move the real cursor onto the middle of the target and hold the hover there.

    Args:
        mgr: The browser manager.
        tid: The tab the cursor is on.
        selector: What the cursor moves onto.

    Returns:
        The point the cursor is held at.
    """
    box = await _element_box(mgr, tid, selector)
    point = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await mgr.move_mouse_to(tid, point[0], point[1])
    await mgr.evaluate(tid, _js_call(_POINTER_FREEZE_JS, _POINTER_EVENTS))
    return point


async def _unpark_pointer(mgr: BrowserManager, tid: str, point: tuple[float, float]) -> None:
    """Let the hover go again and move the real cursor away from the parked point.

    Args:
        mgr: The browser manager.
        tid: The tab the cursor is on.
        point: Where the cursor was held.
    """
    await mgr.evaluate(tid, "() => window.__thawPointer()")
    await mgr.move_mouse_to(tid, point[0], point[1])
    await mgr.move_mouse_to(tid, *_POINTER_AWAY)


_DT_COMPLETED = "e2e-detail-completed"
_DEFAULT_TAB_STATE_JS = """
() => {
  const active = {};
  for (const b of document.querySelectorAll('.td-tabs .td-tab')) active[b.dataset.tab] = b.classList.contains("active");
  const well = document.querySelector('.td-well');
  return {
    active,
    resultOn: !!well && !!well.querySelector('.res-pending, .feedwrap, .res-nores, .res-body-block, .feed-state'),
    logOn: !!document.querySelector('#log-well'),
  };
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("viewport", "status", "expected"),
    [
        (_MOBILE_VIEWPORT, "completed", "res"),
        (_MOBILE_VIEWPORT, "running", "res"),
        (_MOBILE_VIEWPORT, "failed", "res"),
        (_WIDE_DESKTOP_VIEWPORT, "completed", "res"),
        (_WIDE_DESKTOP_VIEWPORT, "running", "res"),
        (_WIDE_DESKTOP_VIEWPORT, "failed", "res"),
    ],
    ids=["mobile-completed", "mobile-running", "mobile-failed",
         "desktop-completed", "desktop-running", "desktop-failed"],
)
@skip_if_no_chrome
async def test_task_detail_default_tab(console_stub: SimpleNamespace, viewport: dict, status: str, expected: str) -> None:
    """The task detail opens on the Results tab on both viewports, whatever the status."""
    tids = {"completed": _DT_COMPLETED, "running": E2E_RUNNING, "failed": E2E_FAILED_250}
    task_id = tids[status]
    await seed_task(_DT_COMPLETED, "rednote_research", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    await seed_task(E2E_RUNNING, "rednote_research")
    await seed_task(E2E_FAILED_250, "rednote_research", E2E_STARTED_AT, "failed", error="boom", ended_at=E2E_ENDED_AT)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        r = await c.get(f"/api/tasks/{task_id}")
        assert r.json()["task"]["status"] == status
    async with e2e_page(viewport) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{task_id}")
        await mgr.wait_for_selector(tab, ".td-tabs .td-tab", timeout=_WAIT_TIMEOUT_MS)
        await _wait_for(
            mgr, tab,
            "() => { const w = document.querySelector('.td-well'); return w && w.children.length > 0 && !w.querySelector('.well-skeleton'); }",
            _WAIT_TIMEOUT_MS)
        state = await mgr.evaluate(tab, _DEFAULT_TAB_STATE_JS)
        assert state["active"]["results"] is (expected == "res")
        assert state["active"]["log"] is (expected == "log")
        assert state["resultOn"] is (expected == "res")
        assert state["logOn"] is (expected == "log")


_STICKY_TID = "e2e-sticky-detail"
_STICKY_RESULT = ("# Sticky e2e\n\n" + ("Filler paragraph to make the page tall. " * 40 + "\n\n") * 60)

_STICKY_STATE_JS = """
() => {
  const screen = document.querySelector(".td-screen");
  const bar = document.querySelector(".td-tabs");
  const toTop = document.querySelector(".td-totop");
  const sep = document.querySelector(".td-totop-sep");
  const cs = getComputedStyle(bar);
  return {
    top: screen.scrollTop,
    position: cs.position,
    stuck: bar.classList.contains("is-stuck"),
    toTopDisplay: toTop ? getComputedStyle(toTop).display : null,
    toTopLabel: toTop ? toTop.getAttribute("aria-label") : null,
    toTopTitle: toTop ? toTop.title : null,
    sepDisplay: sep ? getComputedStyle(sep).display : null,
    shadow: cs.boxShadow,
  };
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("viewport",),
    [(_DESKTOP_VIEWPORT,), (_MOBILE_VIEWPORT,)],
    ids=["desktop", "mobile"],
)
@skip_if_no_chrome
async def test_sticky_bar_toggles_and_back_to_top(console_stub: SimpleNamespace, viewport: dict) -> None:
    """At the top the bar is in flow with the action hidden. Scrolled down it is stuck and raised with the action shown."""
    await seed_task(_STICKY_TID, "rednote_research", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(console_stub.data_dir, _STICKY_TID, "result", _STICKY_RESULT)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        assert (await c.get(f"/api/tasks/{_STICKY_TID}")).json()["task"]["status"] == "completed"
    async with e2e_page(viewport) as (mgr, tab):
        await mgr.navigate(tab, base + f"/task/{_STICKY_TID}")
        await mgr.wait_for_selector(tab, ".td-tabs", timeout=_WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, "() => document.querySelector('.td-screen').scrollHeight > 1500", _WAIT_TIMEOUT_MS)
        # At the top the bar is in flow, sticky, not stuck, the action is hidden, no shadow.
        top = await mgr.evaluate(tab, _STICKY_STATE_JS)
        assert top["position"] == "sticky"
        assert top["stuck"] is False
        assert top["toTopDisplay"] == "none"
        assert top["sepDisplay"] == "none"
        assert top["shadow"] == "none"
        # Scrolled deep the bar is stuck and raised with a shadow, the action is visible and labelled.
        await mgr.evaluate(tab, "() => { const s = document.querySelector('.td-screen'); s.scrollTo({ top: 2000, behavior: 'auto' }); }")
        await _wait_for(mgr, tab, "() => document.querySelector('.td-tabs').classList.contains('is-stuck')", _WAIT_TIMEOUT_MS)
        mid = await mgr.evaluate(tab, _STICKY_STATE_JS)
        assert mid["stuck"] is True
        assert mid["toTopDisplay"] == "flex"
        assert mid["toTopLabel"] == "Back to top"
        assert mid["toTopTitle"] == "Back to top"
        assert mid["sepDisplay"] == "flex"
        assert mid["shadow"] != "none", "stuck bar must carry the raised --shadow"
        # Back to top scrolls the screen to 0 and the bar leaves the stuck state.
        await mgr.evaluate(tab, "() => { document.querySelector('.td-screen').scrollTop = 500; }")
        await mgr.click(tab, ".td-totop")
        await _wait_for(
            mgr, tab,
            "() => document.querySelector('.td-screen').scrollTop === 0 && !document.querySelector('.td-tabs').classList.contains('is-stuck')",
            _WAIT_TIMEOUT_MS,
        )
        back = await mgr.evaluate(tab, _STICKY_STATE_JS)
        assert back["top"] == 0
        assert back["stuck"] is False
        assert back["toTopDisplay"] == "none"


_STICKY_MICROTAB_JS = """
() => {
  const screen = document.querySelector(".td-screen");
  const bar = document.querySelector(".td-tabs");
  const pill = document.querySelector(".td-well .res-microtab-pill");
  const br = bar.getBoundingClientRect();
  const pr = pill.getBoundingClientRect();
  const cx = pr.left + pr.width / 2;
  const cy = pr.top + pr.height / 2;
  const top = document.elementFromPoint(cx, cy);
  return {
    barTop: br.top,
    barBottom: br.bottom,
    pillTop: pr.top,
    pillBottom: pr.bottom,
    inBar: pr.top >= br.top && pr.bottom <= br.bottom,
    hitBar: top ? top.closest(".td-tabs") !== null : false,
    hitClass: top ? top.className : null,
  };
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_stuck_bar_paints_above_result_microtab(console_stub: SimpleNamespace) -> None:
    """On a phone the stuck tab strip paints above the result Preview|Raw chip while they overlap."""
    await seed_task(_STICKY_TID, "rednote_research", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(console_stub.data_dir, _STICKY_TID, "result", _STICKY_RESULT)
    base = console_stub.base_url
    async with e2e_page(_MOBILE_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/task/{_STICKY_TID}")
        await mgr.wait_for_selector(tab, ".td-well .res-microtab-pill", timeout=_WAIT_TIMEOUT_MS)
        await mgr.evaluate(
            tab,
            "() => { const s = document.querySelector('.td-screen'); const chip = document.querySelector('.td-well .res-microtab'); s.scrollTop = chip.getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop; }"
        )
        await _wait_for(mgr, tab, "() => document.querySelector('.td-tabs').classList.contains('is-stuck')", _WAIT_TIMEOUT_MS)
        state = await mgr.evaluate(tab, _STICKY_MICROTAB_JS)
        assert state["inBar"], "precondition, the pill must sit inside the stuck bar band"
        assert state["hitBar"], "the stuck bar must paint above the chip, the hit test returned " + str(state["hitClass"])


# --- Params block tests ---


_P_T8 = "e2e-p-t8"
_P_EMPTY_JS = """
() => {
  const emptyEl = document.querySelector('#p-well .p-empty');
  return {
    rowCount: document.querySelectorAll('#p-well .p-row').length,
    emptyText: emptyEl ? emptyEl.textContent.trim() : null,
    count: document.querySelector('#p-count').textContent,
    copyAllAbsent: !document.querySelector('#p-copy-all'),
  };
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_params_empty_state_zero_rows(console_stub: SimpleNamespace) -> None:
    """With no params to show, the block shows an empty line, hides the count, and has no copy-all icon."""
    await seed_task(_P_T8, "redfin_search", E2E_STARTED_AT, "completed",
                    params={}, ended_at=E2E_ENDED_AT)
    base = console_stub.base_url
    async with e2e_page(_WIDE_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/redfin_search/{_P_T8}")
        await mgr.wait_for_selector(tab, "#p-well .p-empty", timeout=_WAIT_TIMEOUT_MS)
        s = await mgr.evaluate(tab, _P_EMPTY_JS)
        assert s["rowCount"] == 0
        assert s["emptyText"] == "No parameters \u2014 this run stored no params."
        assert s["count"] == ""
        assert s["copyAllAbsent"] is True


_P_T9 = "e2e-p-t9"
_P_NOREC_JS = """
() => {
  return {
    hasParams: !!document.querySelector('.td-params'),
    hasWell: !!document.querySelector('#p-well'),
  };
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_params_null_no_tool_record(console_stub: SimpleNamespace) -> None:
    """A task whose record has no tool renders no Params section at all."""
    await store_module.task_store.insert_running(_P_T9, None, E2E_STARTED_AT, f"runs/{_P_T9}")
    await store_module.task_store.finish(_P_T9, "completed", ended_at=E2E_ENDED_AT)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        assert (await c.get(f"/api/tasks/{_P_T9}")).json()["task"]["params"] is None
    async with e2e_page(_WIDE_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/redfin_search/{_P_T9}")
        await mgr.wait_for_selector(tab, "#d-badge", timeout=_WAIT_TIMEOUT_MS)
        s = await mgr.evaluate(tab, _P_NOREC_JS)
        assert s["hasParams"] is False
        assert s["hasWell"] is False


_RF_TID = "e2e-media-result"
_RF_LIVE = "media_gen/gen_001.png"
_RF_GCED = "media_gen/gen_999.png"
_RF_SETTLE_MS = 30000
_RF_GCED_SIZE = 234567
_RF_GCED_MTIME = int(datetime(2026, 8, 23, 1, 0, 0, tzinfo=timezone.utc).timestamp())
_RF_RESULT_TEXT = (
    "Saved: media_gen/gen_001.png\n"
    "WebDAV download:\n"
    'curl -o gen_001.png "http://127.0.0.1:8315/dav/media_gen/gen_001.png"\n'
    "Saved: media_gen/gen_999.png\n"
    "WebDAV download:\n"
    'curl -o gen_999.png "http://127.0.0.1:8315/dav/media_gen/gen_999.png"\n'
)
_RF_SIDE_CAR = (
    json.dumps({"path": _RF_LIVE, "size": len(E2E_PNG), "mtime": _RF_GCED_MTIME}) + "\n"
    + json.dumps({"path": _RF_GCED, "size": _RF_GCED_SIZE, "mtime": _RF_GCED_MTIME}) + "\n"
)


def _remove_stale_downloads(rel: str) -> None:
    """Remove any downloads left over from a previous run of the same file."""
    for stale in Path(download_dir()).glob(os.path.basename(rel) + "*"):
        stale.unlink(missing_ok=True)


_RF_BLOCKS_JS = """
() => {
  const blocks = Array.from(document.querySelectorAll('.vp-file-block'));
  return blocks.map((b) => {
    const img = b.querySelector('.media-stage img');
    const meta = b.querySelector('.vp-file-meta');
    const title = b.querySelector('.mini-state-title');
    const sub = b.querySelector('.vp-state-sub');
    return {
      path: b.dataset.path,
      present: b.classList.contains('vp-file-present'),
      imgSrc: img ? img.getAttribute('src') : null,
      meta: meta ? meta.textContent : null,
      acts: Array.from(b.querySelectorAll('.vp-file-act')).map((a) => a.getAttribute('href')),
      stateTitle: title ? title.textContent : null,
      stateSub: sub ? sub.textContent : null,
    };
  });
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_result_files_share_the_resource_viewer_mechanism(console_stub: SimpleNamespace) -> None:
    """Result files open in the viewer and remain accessible from Resources."""
    await seed_task(_RF_TID, "generate_image", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(console_stub.data_dir, _RF_TID, "result", _RF_RESULT_TEXT)
    write_run_file(console_stub.data_dir, _RF_TID, "files", _RF_SIDE_CAR)
    write_share_file(console_stub.share, _RF_LIVE, E2E_PNG)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        r = await c.get(f"/api/tasks/{_RF_TID}")
        assert r.json()["task"]["status"] == "completed"
    async with e2e_page(_WIDE_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/generate_image/{_RF_TID}")
        await mgr.wait_for_selector(tab, '.vp-file-block[data-path="media_gen/gen_001.png"]', timeout=_WAIT_TIMEOUT_MS)
        await _wait_for(
            mgr, tab,
            "() => { const b = document.querySelector('.vp-file-block[data-path=\"media_gen/gen_001.png\"]');"
            " return b !== null && b.querySelector('.media-stage img') !== null; }",
            _RF_SETTLE_MS,
        )
        blocks = {block["path"]: block for block in await mgr.evaluate(tab, _RF_BLOCKS_JS)}
        live, gced = blocks[_RF_LIVE], blocks[_RF_GCED]
        assert live["present"] is True
        assert live["imgSrc"] == "/dav/" + _RF_LIVE + "?ephemeral=1"
        assert live["meta"].startswith(str(len(E2E_PNG)) + " B · ")
        assert live["acts"] == ["/view/" + _RF_LIVE, "/resources?select=" + _RF_LIVE, "/dav/dl?path=" + _RF_LIVE]
        assert gced["present"] is False
        assert gced["imgSrc"] is None
        assert gced["meta"] == "229 KB · " + datetime.fromtimestamp(_RF_GCED_MTIME).strftime("%b %d %H:%M")
        assert gced["acts"] == ["/view/" + _RF_GCED, "/resources?select=" + _RF_GCED, "/dav/dl?path=" + _RF_GCED]
        assert gced["stateTitle"] == "No longer in the resource."
        assert gced["stateSub"] == "Garbage-collected (30-day retention)."
        before = mgr.tab_ids()
        await mgr.click(tab, '.vp-file-block[data-path="media_gen/gen_001.png"] .vp-file-act')
        url = await _wait_new_page(mgr, before, base, _WAIT_TIMEOUT_MS)
        assert url == base + "/view/" + _RF_LIVE
        await asyncio.to_thread(_remove_stale_downloads, _RF_LIVE)
        before_dl = mgr.tab_ids()
        dl = await mgr.wait_for_download(
            tab,
            action=lambda: mgr.click(tab, '.vp-file-block[data-path="media_gen/gen_001.png"] .vp-file-act[href^="/dav/dl"]'),
            timeout=_WAIT_TIMEOUT_MS // _MS_PER_S,
        )
        assert dl["downloaded"] is True
        assert await asyncio.to_thread(Path(dl["path"]).read_bytes) == E2E_PNG
        before_urls = [mgr.url(t) for t in before_dl if (mgr.url(t) or "").startswith(base)]
        now_urls = [mgr.url(t) for t in mgr.tab_ids() if (mgr.url(t) or "").startswith(base)]
        assert now_urls == before_urls
        await asyncio.to_thread(Path(dl["path"]).unlink, missing_ok=True)


_FAILED_LOG_TOTAL = 250
_TERMINAL_TOP_VISIBLE_JS = """
(n) => {
  const bar = document.querySelector('.term-top');
  return bar !== null && !bar.classList.contains('hidden') && bar.textContent === 'Show earlier ' + n + ' lines';
}
"""
_TERMINAL_EXPANDED_JS = """
(n) => {
  const bar = document.querySelector('.term-top');
  return bar !== null && !bar.classList.contains('hidden') && bar.textContent === 'Collapse to tail'
    && document.querySelectorAll('.term-lines .term-line').length === n;
}
"""
_TERMINAL_RENDERED_TEXT_JS = """
() => Array.from(document.querySelectorAll('.term-lines .term-line')).map((l) => l.textContent).join('\\n')
"""
_TERMINAL_PILL_SHOWN_JS = "() => document.querySelector('.follow-pill button.show') !== null"
_END_TICK_LAST_LINE_JS = """
() => {
  const lines = document.querySelectorAll('.term-lines .term-line:not(.blank)');
  const last = lines[lines.length - 1];
  return last !== undefined && last.classList.contains('end-tick')
    && document.querySelectorAll('.term-line.end-tick').length === 1;
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_terminal_log_keeps_earlier_bar(console_stub: SimpleNamespace) -> None:
    """A finished task keeps the show earlier bar.

    A click on it grows the log to the whole file, and going back to the tail keeps the
    end mark on the last line.
    """
    await seed_task(E2E_FAILED_250, "rednote_research", E2E_STARTED_AT, "failed", error="boom", ended_at=E2E_ENDED_AT)
    write_run_file(console_stub.data_dir, E2E_FAILED_250, "log", log_lines(_FAILED_LOG_TOTAL))
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        b = (await c.get(f"/api/tasks/{E2E_FAILED_250}/log", params={"lines": 200, "end_offset": 0})).json()
        assert b["total_lines"] == _FAILED_LOG_TOTAL
        assert b["head_reached"] is False
        ground = (await c.get(f"/api/tasks/{E2E_FAILED_250}/log", params={"offset": 0, "max_bytes": 1048576})).json()["content"]
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{E2E_FAILED_250}")
        await mgr.wait_for_selector(tab, ".td-tabs .td-tab", timeout=_WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.td-tab[data-tab="log"]')
        await _wait_for(mgr, tab, _js_call(_TERMINAL_TOP_VISIBLE_JS, _FAILED_LOG_TOTAL - 200), _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#log-count').textContent") == "200 lines"
        assert not await mgr.evaluate(tab, _TERMINAL_PILL_SHOWN_JS)
        await _wait_for(mgr, tab, _END_TICK_LAST_LINE_JS, _WAIT_TIMEOUT_MS)
        await mgr.evaluate(tab, "() => document.querySelector('.term-top').click()")
        await _wait_for(mgr, tab, _js_call(_TERMINAL_EXPANDED_JS, _FAILED_LOG_TOTAL), _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#log-count').textContent") == f"{_FAILED_LOG_TOTAL} lines"
        assert not await mgr.evaluate(tab, _TERMINAL_PILL_SHOWN_JS)
        assert await mgr.evaluate(tab, _TERMINAL_RENDERED_TEXT_JS) == ground.rstrip("\n")
        assert await mgr.evaluate(tab, _END_TICK_LAST_LINE_JS)
        await mgr.evaluate(tab, "() => document.querySelector('.term-top').click()")
        await _wait_for(mgr, tab, _js_call(_TERMINAL_TOP_VISIBLE_JS, _FAILED_LOG_TOTAL - 200), _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#log-count').textContent") == "200 lines"
        await _wait_for(mgr, tab, _END_TICK_LAST_LINE_JS, _WAIT_TIMEOUT_MS)


_LEAK_MARKER_LINE = "====================== leak chunk marker ======================"
# An earlier chunk with a block marker, plus the tail the screen reads first.
_LEAK_LOG_EARLIER = (
    "".join(f"leak-old {i:03d}\n" for i in range(4)) + _LEAK_MARKER_LINE + "\n"
    + "".join(f"leak-old {i:03d}\n" for i in range(4, 9))
)
_LEAK_LOG_TAIL = "".join(f"leak-tail {i:03d}\n" for i in range(200))
_LEAK_LOG_INITIAL = _LEAK_LOG_EARLIER + _LEAK_LOG_TAIL
_LEAK_APPEND_LINE = "leak-follow 00\n"
_LEAK_BAR_VISIBLE_JS = """
() => {
  const bar = document.querySelector('.term-top');
  return bar !== null && !bar.classList.contains('hidden');
}
"""
_LEAK_MARKER_PAINTED_JS = """
() => Array.from(document.querySelectorAll('.term-lines .term-line')).some((l) => l.textContent.includes('leak chunk marker'))
"""
_LEAK_FOLLOW_COUNT_JS = """
() => Array.from(document.querySelectorAll('.term-lines .term-line')).filter((l) => l.textContent.startsWith('leak-follow')).length
"""
_LEAK_FOLLOW_GROWN_JS = """
(before) => Array.from(document.querySelectorAll('.term-lines .term-line'))
  .filter((l) => l.textContent.startsWith('leak-follow')).length > before
"""
_LEAK_LAST_JS = """
() => {
  const lines = Array.from(document.querySelectorAll('.term-lines .term-line'))
    .filter((l) => l.textContent.startsWith('leak-follow'));
  const last = lines[lines.length - 1];
  return { block: last.classList.contains('block'), cls: last.className };
}
"""


def _append_leak_line(path: Path) -> None:
    """Add one forward-follow line to a log file."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(_LEAK_APPEND_LINE)


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_show_earlier_chunk_does_not_leak_block_state(console_stub: SimpleNamespace) -> None:
    """Loading earlier log lines does not change how the next new line is shown."""
    await seed_task(E2E_RUNNING, "rednote_research")
    log_path = write_run_file(console_stub.data_dir, E2E_RUNNING, "log", _LEAK_LOG_INITIAL)
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{E2E_RUNNING}")
        await mgr.wait_for_selector(tab, ".td-tabs .td-tab", timeout=_WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.td-tab[data-tab="log"]')
        await _wait_for(mgr, tab, _LEAK_BAR_VISIBLE_JS, _WAIT_TIMEOUT_MS)
        await mgr.evaluate(tab, "() => document.querySelector('.term-top').click()")
        await _wait_for(mgr, tab, _LEAK_MARKER_PAINTED_JS, _WAIT_TIMEOUT_MS)
        before = await mgr.evaluate(tab, _LEAK_FOLLOW_COUNT_JS)
        await asyncio.to_thread(_append_leak_line, log_path)
        await _wait_for(mgr, tab, _js_call(_LEAK_FOLLOW_GROWN_JS, before), _WAIT_TIMEOUT_MS)
        last = await mgr.evaluate(tab, _LEAK_LAST_JS)
        assert last["block"] is False


_FLIP_TID = E2E_RUNNING
_FLIP_FIRST_PAINT_DELAY_S = 4.0
# The log is longer than the 200 lines the first read shows.
# The end mark's forward read must land on a tail line.
_FLIP_BODY_LINES = 240
_FLIP_TAIL_LINES = 10
_FLIP_SHOWN_LINES = 200
_FLIP_LOG = "".join(f"flip-body {i:03d}\n" for i in range(_FLIP_BODY_LINES)) + \
    "".join(f"flip-tail {i:03d}\n" for i in range(_FLIP_TAIL_LINES))

_FLIP_TICK_ON_TAIL_JS = """
() => {
  const lines = document.querySelectorAll('.term-lines .term-line:not(.blank)');
  const last = lines[lines.length - 1];
  return last !== undefined && last.textContent.startsWith('flip-tail')
    && last.classList.contains('end-tick')
    && document.querySelectorAll('.term-line.end-tick').length === 1;
}
"""
_FLIP_COUNT_JS = "(n) => document.querySelector('#log-count').textContent === (n + ' lines')"


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_terminal_flip_during_first_paint_keeps_single_end_tick(console_stub: SimpleNamespace) -> None:
    """A task that finishes while the log is still loading ends up with one end mark."""
    await seed_task(_FLIP_TID, "rednote_research")
    write_run_file(console_stub.data_dir, _FLIP_TID, "log", _FLIP_LOG)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        assert (await c.get(f"/api/tasks/{_FLIP_TID}")).json()["task"]["status"] == "running"
    # Hold the first log read so the task finishes while the first paint is still loading.
    console_stub.delay(f"/tasks/{_FLIP_TID}/log", _FLIP_FIRST_PAINT_DELAY_S, first_only=True)
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{_FLIP_TID}")
        await mgr.wait_for_selector(tab, ".td-tabs .td-tab", timeout=_WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.td-tab[data-tab="log"]')
        now_iso = (datetime.now(timezone.utc) + timedelta(milliseconds=100)).isoformat()
        await console_stub.finish(_FLIP_TID, "failed", error="stub failure", ended_at=now_iso)
        await _wait_for(mgr, tab, _FLIP_TICK_ON_TAIL_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#d-badge').textContent") == "failed"
        assert await mgr.evaluate(tab, _js_call(_FLIP_COUNT_JS, _FLIP_SHOWN_LINES))
        assert not await mgr.evaluate(tab, _TERMINAL_PILL_SHOWN_JS)


_FLIP_SUBTASK_EXTRA = "e2e-extra-child"
_SUBTASKS_TWO_ROWS_JS = "() => document.querySelectorAll('#sub-well .st-row').length === 2"
_SUBTASKS_THREE_ROWS_JS = "() => document.querySelectorAll('#sub-well .st-row').length === 3"


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_terminal_flip_repaints_subtasks_with_delta_missed_child(console_stub: SimpleNamespace) -> None:
    """When a task finishes, a subtask left out of the update still shows up without leaving the page."""
    await seed_task(_FLIP_TID, "rednote_research")
    await seed_task("e2e-running-sub-1", "rednote_research", "2026-08-23T01:00:01+00:00", parent_task_id=_FLIP_TID)
    await seed_task("e2e-running-sub-2", "rednote_research", "2026-08-23T01:00:02+00:00", parent_task_id=_FLIP_TID)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        real = (await c.get(f"/api/tasks/{_FLIP_TID}")).json()
    assert real["task"]["status"] == "running"
    assert len(real["subtasks"]) == 2
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{_FLIP_TID}")
        await mgr.wait_for_selector(tab, ".td-tabs .td-tab", timeout=_WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.td-tab[data-tab="subtasks"]')
        await _wait_for(mgr, tab, _SUBTASKS_TWO_ROWS_JS, _WAIT_TIMEOUT_MS)
        # A child that ends while the parent is still running shows up from the changes feed.
        await seed_task(_FLIP_SUBTASK_EXTRA, "web_fetch", E2E_STARTED_AT, "completed", parent_task_id=_FLIP_TID)
        now_iso = (datetime.now(timezone.utc) + timedelta(milliseconds=100)).isoformat()
        await console_stub.finish(_FLIP_TID, "failed", error="stub failure", ended_at=now_iso)
        await _wait_for(mgr, tab, "() => document.querySelector('#d-badge').textContent === 'failed'", _WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, _SUBTASKS_THREE_ROWS_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(
            tab, "() => document.querySelector('.td-tab[data-tab=\"subtasks\"]').textContent") == "Subtasks 3"
        assert await mgr.evaluate(
            tab, "() => document.querySelector('#sub-well .st-row[data-task-id=\"e2e-extra-child\"]') !== null")


_REFRESH_TID = "e2e-refresh-parent"
_REFRESH_CHILD = "e2e-refresh-child"


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_subtask_row_appears_while_parent_still_running(console_stub: SimpleNamespace) -> None:
    """A child delivered by the changes poll shows up in the subtask list
    while the parent run is still running, without a reload or the parent
    finishing, and the tab count label follows it."""
    await seed_task(_REFRESH_TID, "rednote_research")
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{_REFRESH_TID}")
        await mgr.wait_for_selector(tab, ".td-tabs .td-tab", timeout=_WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.td-tab[data-tab="subtasks"]')
        # The empty snapshot renders the empty card at mount.
        await _wait_for(
            mgr, tab, "() => document.querySelector('#sub-well .st-empty') !== null", _WAIT_TIMEOUT_MS)
        await seed_task(_REFRESH_CHILD, "web_fetch", E2E_STARTED_AT, parent_task_id=_REFRESH_TID)
        await _wait_for(
            mgr, tab,
            f"() => document.querySelector('#sub-well .st-row[data-task-id=\"{_REFRESH_CHILD}\"]') !== null",
            _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#d-badge').textContent") == "running"
        assert await mgr.evaluate(
            tab, "() => document.querySelector('.td-tab[data-tab=\"subtasks\"]').textContent") == "Subtasks 1"


_NOT_FOUND_VIEW_JS = """
() => {
  const e = document.querySelector('#view .empty');
  return e !== null && e.textContent.includes('This screen does not exist');
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_malformed_percent_escape_route_is_not_found(console_stub: SimpleNamespace) -> None:
    """A path with a broken percent escape shows the not found screen."""
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, console_stub.base_url + "/tools/100%discount")
        await _wait_for(mgr, tab, _NOT_FOUND_VIEW_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#view .empty .t-title').textContent") == "Not found"
        text = await mgr.evaluate(tab, "() => document.getElementById('view').innerText")
        assert "This screen does not exist" in text
        assert "Back to Tools" in text


_ROOT_TOOLS_RENDERED_JS = """
() => location.pathname === '/tools' && document.querySelector('#l1-list .l1-row') !== null
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_root_path_normalizes_to_tools_and_nav_survives(console_stub: SimpleNamespace) -> None:
    """Going back to / shows the Tools screen and later clicks still work."""
    write_share_file(console_stub.share, "downloads/e2e-seed.txt", b"seed")
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, console_stub.base_url + "/tools")
        await _wait_for(mgr, tab, "() => document.querySelector('#l1-list .l1-row') !== null", _WAIT_TIMEOUT_MS)
        await mgr.evaluate(tab, "() => { history.pushState(null, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')); }")
        await _wait_for(mgr, tab, _ROOT_TOOLS_RENDERED_JS, _WAIT_TIMEOUT_MS)
        await mgr.click(tab, 'a[data-spa][data-nav="resources"]')
        await _wait_for(mgr, tab, "() => document.querySelector('#r-tree .t-row') !== null", _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => location.pathname") == "/resources"


_SEARCH_ROOT = "e2e_search_root"
_SEARCH_MID = _SEARCH_ROOT + "/e2e_search_mid"
_SEARCH_MATCH = _SEARCH_MID + "/e2e_search_match"
_CHAIN_PATHS = [_SEARCH_ROOT, _SEARCH_MID, _SEARCH_MATCH]

_MATCH_ROW_JS = """
() => {
  const rows = Array.from(document.querySelectorAll('#r-tree .t-row.folder:not(.dimmed)'));
  return rows.some((r) => r.querySelector('.t-name') && r.querySelector('.t-name').textContent === 'e2e_search_match/');
}
"""

_CLICK_MATCH_JS = """
() => {
  const rows = Array.from(document.querySelectorAll('#r-tree .t-row.folder:not(.dimmed)'));
  const m = rows.find((r) => r.querySelector('.t-name').textContent === 'e2e_search_match/');
  m.click();
}
"""

_CHAIN_READY_JS = """
(paths) => paths.every((p) => {
  const box = document.querySelector('.t-children[data-children="' + p + '"]');
  return box && box.classList.contains('open') && box.querySelector(':scope > .t-row') !== null;
})
"""

_CHAIN_STATE_JS = """
(paths) => {
  const out = {};
  for (const p of paths) {
    const box = document.querySelector('.t-children[data-children="' + p + '"]');
    out[p] = {
      open: box ? box.classList.contains('open') : false,
      children: box ? Array.from(box.querySelectorAll(':scope > .t-row')).map((r) => r.dataset.path) : null,
    };
  }
  return out;
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_resources_search_dir_click_expands_chain(console_stub: SimpleNamespace) -> None:
    """Clicking a folder found by search opens it and every folder above it."""
    write_share_file(console_stub.share, _SEARCH_MATCH + "/alpha.txt", b"e2e search click")
    write_share_file(console_stub.share, _SEARCH_MATCH + "/beta.txt", b"e2e search click")
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        body = (await c.get("/api/resources")).json()
        assert _SEARCH_ROOT in [e["name"] for e in body["entries"]]
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/resources")
        await mgr.wait_for_selector(tab, "#r-tree .t-row.folder")
        await mgr.type_text(tab, "#r-search", "e2e_search_match")
        await _wait_for(mgr, tab, _MATCH_ROW_JS, _WAIT_TIMEOUT_MS)
        await mgr.evaluate(tab, _CLICK_MATCH_JS)
        await _wait_for(mgr, tab, _js_call(_CHAIN_READY_JS, _CHAIN_PATHS), _WAIT_TIMEOUT_MS)
        state = await mgr.evaluate(tab, _js_call(_CHAIN_STATE_JS, _CHAIN_PATHS))
        assert state[_SEARCH_ROOT]["open"] is True
        assert _SEARCH_MID in state[_SEARCH_ROOT]["children"]
        assert state[_SEARCH_MID]["open"] is True
        assert _SEARCH_MATCH in state[_SEARCH_MID]["children"]
        assert state[_SEARCH_MATCH]["open"] is True
        assert _SEARCH_MATCH + "/alpha.txt" in state[_SEARCH_MATCH]["children"]
        assert _SEARCH_MATCH + "/beta.txt" in state[_SEARCH_MATCH]["children"]
        assert await mgr.evaluate(tab, "() => document.querySelector('#r-search').value") == ""


_CRUMB_ROOT = "e2e_crumb_root"
_CRUMB_MID = _CRUMB_ROOT + "/e2e_crumb_mid"
_CRUMB_FILE = _CRUMB_MID + "/note.txt"

# The settled search view shows the dimmed parent folders plus the matching file row, with no non-dimmed top-level folder.
_CRUMB_SEARCH_SETTLED_JS = """
() => document.querySelector('#r-tree > .t-row.dimmed') !== null
       && document.querySelector('#r-tree > .t-row.file') !== null
       && document.querySelector('#r-tree > .t-row.folder:not(.dimmed)') === null
"""

_CRUMB_LIVE_TREE_JS = """
() => document.querySelector('#r-tree > .t-row.folder:not(.dimmed)') !== null && document.querySelectorAll('#r-crumbs .seg').length === 0
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_resources_crumb_click_clears_active_search(console_stub: SimpleNamespace) -> None:
    """Clicking a crumb during a search brings back the tree and the arrow keys work again."""
    write_share_file(console_stub.share, _CRUMB_FILE, b"e2e crumb click")
    write_share_file(console_stub.share, _CRUMB_MID + "/capture_e2e.png", E2E_PNG)
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/resources")
        await mgr.wait_for_selector(tab, "#r-tree .t-row.folder")
        await mgr.click(tab, '.t-row.folder[data-path="e2e_crumb_root"]')
        await _wait_for(mgr, tab, _js_call(_CHAIN_READY_JS, [_CRUMB_ROOT]), _WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.t-children[data-children="e2e_crumb_root"] > .t-row')
        await _wait_for(mgr, tab, _js_call(_CHAIN_READY_JS, [_CRUMB_MID]), _WAIT_TIMEOUT_MS)
        await mgr.click(tab, f'.t-row.file[data-path="{_CRUMB_FILE}"]')
        await _wait_for(mgr, tab, "() => document.querySelectorAll('#r-crumbs .seg').length === 2", _WAIT_TIMEOUT_MS)
        await mgr.type_text(tab, "#r-search", "capture")
        await _wait_for(mgr, tab, _CRUMB_SEARCH_SETTLED_JS, _WAIT_TIMEOUT_MS)
        await mgr.click(tab, "#r-crumbs span:nth-of-type(3)")
        await _wait_for(mgr, tab, _CRUMB_LIVE_TREE_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#r-search').value") == ""
        await mgr.press_key(tab, "ArrowDown")
        await _wait_for(mgr, tab, "() => document.querySelector('#r-tree .t-row.file.selected') !== null", 5000)


_INDENT_ROOT = "e2e_indent_root"
_INDENT_MID = _INDENT_ROOT + "/e2e_indent_mid"

_INDENT_PADS_JS = """
(paths) => ({
  d0: getComputedStyle(document.querySelector('#r-tree > .t-row')).paddingLeft,
  d1: getComputedStyle(document.querySelector('.t-children[data-children="' + paths[0] + '"] > .t-row')).paddingLeft,
  d2: getComputedStyle(document.querySelector('.t-children[data-children="' + paths[1] + '"] > .t-row')).paddingLeft,
})
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("viewport", "expected"),
    [
        (_DESKTOP_VIEWPORT, {"d0": "8px", "d1": "20px", "d2": "32px"}),
        (_MOBILE_VIEWPORT, {"d0": "8px", "d1": "30px", "d2": "60px"}),
    ],
    ids=["desktop", "mobile"],
)
@skip_if_no_chrome
async def test_resources_tree_indent(console_stub: SimpleNamespace, viewport: dict, expected: dict) -> None:
    """Tree rows step further right at each level, and the step is wider on a phone."""
    write_share_file(console_stub.share, _INDENT_MID + "/leaf.txt", b"e2e indent")
    base = console_stub.base_url
    async with e2e_page(viewport) as (mgr, tab):
        await mgr.navigate(tab, base + "/resources")
        await mgr.wait_for_selector(tab, "#r-tree .t-row.folder")
        await mgr.click(tab, '.t-row.folder[data-path="e2e_indent_root"] .t-caret')
        await _wait_for(mgr, tab, _js_call(_CHAIN_READY_JS, [_INDENT_ROOT]), _WAIT_TIMEOUT_MS)
        await mgr.click(tab, '.t-children[data-children="e2e_indent_root"] > .t-row .t-caret')
        await _wait_for(mgr, tab, _js_call(_CHAIN_READY_JS, [_INDENT_MID]), _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _js_call(_INDENT_PADS_JS, [_INDENT_ROOT, _INDENT_MID])) == expected


_WELL_DOC = "e2e_well_plan.md"

_WELL_FIT_JS = """
() => {
  const box = document.querySelector(".pv-well").getBoundingClientRect();
  return { x: box.x, right: box.right, inner: window.innerWidth };
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "viewport",
    [_MOBILE_VIEWPORT, _WIDE_DESKTOP_VIEWPORT],
    ids=["mobile", "desktop"],
)
@skip_if_no_chrome
async def test_resources_preview_well_fits_viewport(console_stub: SimpleNamespace, viewport: dict) -> None:
    """A selected doc's preview well stays inside the viewport, right edge included."""
    write_share_file(console_stub.share, _WELL_DOC, b"# e2e plan\n\nThe plan body.\n")
    base = console_stub.base_url
    async with e2e_page(viewport) as (mgr, tab):
        await mgr.navigate(tab, base + "/resources?select=" + _WELL_DOC)
        await mgr.wait_for_selector(tab, ".pv-well .md-doc", timeout=_WAIT_TIMEOUT_MS)
        s = await mgr.evaluate(tab, _WELL_FIT_JS)
        assert s["x"] >= 0, f".pv-well left edge is off the viewport: {s}"
        assert s["right"] - s["inner"] <= 1, (
            f".pv-well overflows the viewport: x={s['x']} right={s['right']} innerWidth={s['inner']}"
        )


_TRUNC_ROOT = "e2e_trunc_root"
_TRUNC_FILE_COUNT = 205  # more than the 200 result search cap, so a search is cut short

_TRUNC_NOTE_JS = """
() => {
  const note = document.querySelector('#r-tree .t-trunc');
  return note !== null && note.textContent === 'Results truncated';
}
"""

_TRUNC_UNTRUNCATED_JS = """
() => {
  const note = document.querySelector('#r-tree .t-trunc');
  const row = Array.from(document.querySelectorAll('#r-tree .t-row.file'))
    .some((r) => (r.dataset.path || '').endsWith('/t_000.txt'));
  return note === null && row;
}
"""


def _seed_trunc_files(root: Path) -> None:
    """Make enough small files under root that a search has to cut its results short."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(_TRUNC_FILE_COUNT):
        (root / f"t_{i:03d}.txt").write_text("e2e trunc\n", encoding="utf-8")


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_resources_search_truncated_note(console_stub: SimpleNamespace) -> None:
    """A search that is cut short shows a note under the results, a full one does not."""
    _seed_trunc_files(console_stub.share / _TRUNC_ROOT)
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/resources")
        await mgr.wait_for_selector(tab, "#r-tree .t-row.folder")
        await mgr.type_text(tab, "#r-search", "e2e_trunc")
        await _wait_for(mgr, tab, _TRUNC_NOTE_JS, 15000)
        await mgr.type_text(tab, "#r-search", "t_000")
        await _wait_for(mgr, tab, _TRUNC_UNTRUNCATED_JS, 15000)


_L2_FETCH_ERROR_JS = """
() => {
  const e = document.querySelector('#tt-list .empty');
  return e !== null && e.textContent.includes('Cannot load tasks');
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_initial_fetch_error_state(console_stub: SimpleNamespace) -> None:
    """When the task list fails to load, the screen shows an error, not an empty list."""
    console_stub.force_status("/tools/rednote_research/tasks", 500, code="internal", message="boom")
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, console_stub.base_url + "/tools/rednote_research")
        await _wait_for(mgr, tab, _L2_FETCH_ERROR_JS, _WAIT_TIMEOUT_MS)
        list_text = await mgr.evaluate(tab, "() => document.querySelector('#tt-list').textContent")
        assert "No runs yet" not in list_text
        assert "Load tasks failed" in await mgr.evaluate(tab, "() => document.querySelector('#toast-stack').textContent")


_DETAIL_FETCH_ERROR_JS = """
() => {
  const t = document.querySelector('#toast-stack .toast');
  return t !== null && t.textContent.includes('Load task failed');
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_task_detail_initial_fetch_error_state(console_stub: SimpleNamespace) -> None:
    """When the task page fails to load, the screen shows an error and a message."""
    await seed_task(E2E_FAILED_250, "rednote_research", E2E_STARTED_AT, "failed", error="boom", ended_at=E2E_ENDED_AT)
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        assert (await c.get(f"/api/tasks/{E2E_FAILED_250}")).status_code == 200
    console_stub.force_status(f"/tasks/{E2E_FAILED_250}", 500, code="internal", message="boom")
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + f"/tools/rednote_research/{E2E_FAILED_250}")
        await _wait_for(mgr, tab, _DETAIL_FETCH_ERROR_JS, _WAIT_TIMEOUT_MS)
        assert "Load task failed" in await mgr.evaluate(tab, "() => document.querySelector('#toast-stack').textContent")
        assert await mgr.evaluate(tab, "() => document.querySelector('#d-badge') === null")
        assert await mgr.evaluate(tab, "() => document.querySelector('.td-tabs') === null")
        assert "Loading task" not in await mgr.evaluate(tab, "() => document.querySelector('#view').textContent")


_ABOUT_STATE_JS = """
() => {
  const el = document.querySelector('.rr-about');
  if (!el) return null;
  const col = document.querySelector('.rr-col');
  const idx = (n) => Array.from(col.children).indexOf(n);
  const btn = el.querySelector('.chev');
  const body = el.querySelector('.rr-about-body');
  const lns = Array.from(el.querySelectorAll('.rr-about-pre .ln'));
  const first = lns[0] || document.createElement('div');
  return {
    label: el.querySelector('.rr-about-lbl').textContent,
    closed: el.classList.contains('closed'),
    aria: btn.getAttribute('aria-expanded'),
    title: btn.getAttribute('title'),
    bodyVisible: getComputedStyle(body).display !== 'none',
    lead: el.querySelector('.rr-about-lead') ? el.querySelector('.rr-about-lead').textContent : null,
    lines: lns.map((l) => l.textContent),
    firstNm: first.querySelector('.nm') ? first.querySelector('.nm').textContent : null,
    firstRst: first.querySelector('.rst') ? first.querySelector('.rst').textContent : null,
    order: {
      about: idx(el),
      form: idx(document.querySelector('#rr-form')),
    },
  };
}
"""


async def _collapse_about(mgr: BrowserManager, tab: str) -> dict:
    """Close the About block, check it is closed, and give back its state."""
    await mgr.evaluate(tab, "() => document.querySelector('.rr-about .chev').click()")
    await _wait_for(mgr, tab, "() => document.querySelector('.rr-about').classList.contains('closed')", _WAIT_TIMEOUT_MS)
    closed = await mgr.evaluate(tab, _ABOUT_STATE_JS)
    assert closed["closed"] is True and closed["aria"] == "false" and closed["title"] == "Expand"
    assert closed["bodyVisible"] is False
    return closed


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_run_about_block_desktop_default_open(console_stub: SimpleNamespace) -> None:
    """On desktop the About block is open, sits between the intro and the form, and the arrow closes it."""
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        r = await c.get("/api/tools/github/schema")
        assert r.status_code == 200 and "Actions:" in r.json()["description"]
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/github/run")
        await mgr.wait_for_selector(tab, ".rr-about-pre .ln", timeout=_WAIT_TIMEOUT_MS)
        state = await mgr.evaluate(tab, _ABOUT_STATE_JS)
        assert state["label"] == "About"
        assert state["closed"] is False and state["aria"] == "true" and state["title"] == "Collapse"
        assert state["order"]["about"] < state["order"]["form"]
        assert state["lead"] == "Actions:"
        assert len(state["lines"]) == 8
        assert state["lines"][0] == "  list_issues    List recent issues in one repo matching a query."
        assert "  view_file      File content (or directory listing) at a path in a repo. " \
               "limits: files up to 100MB" in state["lines"]
        assert state["bodyVisible"] is True
        await _collapse_about(mgr, tab)


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_run_about_block_phone_default_open_reflow(console_stub: SimpleNamespace) -> None:
    """On a phone the About block also starts open.

    Closing it hides the text, and each entry reads as a name followed by what it does.
    """
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        r = await c.get("/api/tools/github/schema")
        assert r.status_code == 200 and len(r.json()["description"].split("\n")) == 14
    async with e2e_page(_MOBILE_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/github/run")
        await mgr.wait_for_selector(tab, ".rr-about-pre .ln", timeout=_WAIT_TIMEOUT_MS)
        state = await mgr.evaluate(tab, _ABOUT_STATE_JS)
        assert state["closed"] is False and state["aria"] == "true" and state["title"] == "Collapse"
        assert state["bodyVisible"] is True
        assert state["lines"][0] == "list_issues: List recent issues in one repo matching a query."
        assert state["firstNm"] == "list_issues"
        assert state["firstRst"] == ": List recent issues in one repo matching a query."
        await _collapse_about(mgr, tab)


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(("path", "tool", "status"),
                         [("/tools/task/run", "task", 403),
                          ("/tools/no_such_tool_xyz/run", "no_such_tool_xyz", 404)])
@skip_if_no_chrome
async def test_run_about_block_absent_on_error_states(
    console_stub: SimpleNamespace, path: str, tool: str, status: int
) -> None:
    """A run page for a hidden or unknown tool shows the empty state and no About block."""
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        r = await c.get(f"/api/tools/{tool}/schema")
        assert r.status_code == status
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + path)
        await mgr.wait_for_selector(tab, ".empty", timeout=_WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _ABOUT_STATE_JS) is None


_L2_SKELETON_JS = """
() => {
  const list = document.querySelector('#tt-list');
  const kids = [...list.children].map((k) =>
    k.id === 'g-hist' ? 'g-hist'
    : k.classList.contains('skel') ? 'skel:' + k.children.length
    : 'other').join(',');
  return kids === 'g-hist,skel:4';
}
"""

_L2_T1 = "aaaa0001-0000-0000-0000-000000000001"
_L2_T2 = "aaaa0002-0000-0000-0000-000000000002"
_L2_T3 = "aaaa0003-0000-0000-0000-000000000003"
_L2_LIVE_TASK = "aaaa0009-0000-0000-0000-000000000009"


def _l2_task(task_id: str, status: str, started_at: str, *, ended_at: str | None = None) -> dict:
    """A task row in the shape the task list screen reads from the api."""
    return {"task_id": task_id, "tool": "rednote_research", "status": status,
            "started_at": started_at, "started_at_local": started_at,
            "ended_at": ended_at, "ended_at_local": ended_at, "params": {"query": "stub"}}


# Two fake pages the per-cursor tasks knob serves, so a list of 200-or-fewer runs still paginates.
_L2_STUB_PAGE1 = [
    _l2_task(_L2_T1, "running", "2026-08-24T08:00:00+00:00"),
    _l2_task(_L2_T2, "completed", "2026-08-24T07:00:00+00:00", ended_at="2026-08-24T07:00:02+00:00"),
    _l2_task(_L2_T3, "failed", "2026-08-24T06:00:00+00:00", ended_at="2026-08-24T06:00:01+00:00"),
]
_L2_STUB_PAGE2 = [
    _l2_task("aaaa0004-0000-0000-0000-000000000004", "completed", "2026-08-24T05:00:00+00:00",
             ended_at="2026-08-24T05:00:02+00:00"),
    _l2_task("aaaa0005-0000-0000-0000-000000000005", "completed", "2026-08-24T04:00:00+00:00",
             ended_at="2026-08-24T04:00:02+00:00"),
]


def _l2_page_body(tasks: list[dict], next_cursor: str | None) -> dict:
    """Build a tasks-list reply with the page rows, the description, and the stamps the screen reads."""
    return {"description": "Research a topic on Xiaohongshu.", "tasks": tasks, "next_cursor": next_cursor,
            "generated_at": "2026-08-24T09:00:00+00:00", "generated_at_local": "2026-08-24 09:00:00"}


def _l2_pages(scope: dict) -> dict:
    """Serve the first page for a cursor-less call and the older page once a cursor is sent."""
    if b"cursor=" in scope.get("query_string", b""):
        return _l2_page_body(_L2_STUB_PAGE2, None)
    return _l2_page_body(_L2_STUB_PAGE1, "stub-cursor")


async def _l2_seed() -> None:
    await seed_task(_L2_T1, "rednote_research")
    await seed_task(_L2_T2, "rednote_research", "2026-08-24T07:00:00+00:00", "completed",
                    ended_at="2026-08-24T07:00:02+00:00")
    await seed_task(_L2_T3, "rednote_research", "2026-08-24T06:00:00+00:00", "failed",
                    ended_at="2026-08-24T06:00:01+00:00")


_L2_SKELETON_GONE_JS = "() => !document.querySelector('#tt-list .skel-row')"
_L2_FADE_COUNT_JS = "() => document.querySelectorAll('#tt-list .t-row.fade-in').length"
_L2_FADE_ONE_JS = "() => document.querySelectorAll('#tt-list .t-row.fade-in').length === 1"
_L2_FADED_ID_JS = "() => document.querySelector('#tt-list .t-row.fade-in .c-id').textContent"
_L2_ROW_COUNT_JS = "() => document.querySelectorAll('#tt-list .t-row').length"
_L2_ROW_IDS_JS = "() => [...document.querySelectorAll('#tt-list .t-row .c-id')].map((e) => e.textContent)"
_L2_LOAD_TEXT_JS = ("(n) => document.querySelector('#tt-load-btn').textContent === "
                    "'Showing most recent ' + n + ' — Load older'")
_L2_LOAD_HIDDEN_JS = "() => document.querySelector('#tt-load').classList.contains('hidden')"


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_skeleton_fade_and_load_older_count(console_stub: SimpleNamespace) -> None:
    """Each group shows a loading block, only a new live row fades in, and the load older count keeps up."""
    console_stub.pin_dynamic("/rednote_research/tasks", _l2_pages)
    console_stub.delay("/rednote_research/tasks", 1.2, first_only=True)
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")
        await _wait_for(mgr, tab, _L2_SKELETON_JS, _WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, _L2_SKELETON_GONE_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_FADE_COUNT_JS) == 0
        assert await mgr.evaluate(tab, _js_call(_L2_LOAD_TEXT_JS, 3))
        assert not await mgr.evaluate(tab, _L2_LOAD_HIDDEN_JS)
        await mgr.evaluate(tab, "() => document.querySelector('#tt-load-btn').click()")
        await _wait_for(mgr, tab, _js_call(_L2_LOAD_TEXT_JS, 5), _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_ROW_COUNT_JS) == 5
        assert await mgr.evaluate(tab, _L2_LOAD_HIDDEN_JS)
        await seed_task(_L2_LIVE_TASK, "rednote_research")
        await _wait_for(mgr, tab, _L2_FADE_ONE_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_FADED_ID_JS) == "aaaa0009"
        assert await mgr.evaluate(tab, _js_call(_L2_LOAD_TEXT_JS, 6))


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_run_group_hidden_while_loading(console_stub: SimpleNamespace) -> None:
    """The Running header stays hidden while the list loads and shows once a running task arrives."""
    await _l2_seed()
    base = console_stub.base_url
    console_stub.delay("/rednote_research/tasks", 1.2, first_only=True)
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")
        await mgr.wait_for_selector(tab, "#tt-list", timeout=_WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, _L2_SKELETON_JS, _WAIT_TIMEOUT_MS)
        # The Running header must not show while the list is still loading.
        assert await mgr.evaluate(tab, _L2_RUN_GROUP_HIDDEN_JS)
        await _wait_for(mgr, tab, _L2_SKELETON_GONE_JS, _WAIT_TIMEOUT_MS)
        assert not await mgr.evaluate(tab, _L2_RUN_GROUP_HIDDEN_JS)
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 1", "· 2"]


_L2_HOVER_ROW_JS = (
    "(i) => { const r = document.querySelectorAll('#tt-list .t-row')[i];"
    " r.dispatchEvent(new MouseEvent('mouseenter', {bubbles: false}));"
    " return r.querySelector('.c-id').textContent; }"
)
_L2_CLICK_CHIP_JS = "(v) => document.querySelector('#tt-chips .chip[data-v=\"' + v + '\"]').click()"
_L2_GROUPS_JS = """
() => {
  const out = { running: [], history: [] };
  let section = "running";
  for (const k of document.querySelector("#tt-list").children) {
    if (k.id === "g-hist") { section = "history"; continue; }
    if (k.classList.contains("t-row")) out[section].push(k.dataset.tid);
  }
  return out;
}
"""
_L2_GROUP_COUNTS_JS = "() => [document.querySelector('#n-run').textContent, document.querySelector('#n-hist').textContent]"
_L2_BADGE_COMPLETED_JS = """
(id) => document.querySelector('#tt-list .t-row[data-tid="' + id + '"] .badge').textContent === "completed"
"""
_L2_POINTER_ON_JS = """
(id) => {
  const r = document.querySelector('#tt-list .t-row[data-tid="' + id + '"]');
  const b = r.getBoundingClientRect();
  return document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2).closest(".t-row") === r;
}
"""
_L2_RUN_COUNT_ZERO_JS = "() => document.querySelector('#n-run').textContent === '· 0'"
_L2_RUN_GROUP_HIDDEN_JS = "() => document.querySelector('#g-run').style.display === 'none'"


async def _l2_trio_seed() -> None:
    """Seed one running task and two history rows, the running one the newest."""
    await seed_task(_L2_T1, "rednote_research", "2026-08-24T08:00:00+00:00")
    await seed_task(_L2_T2, "rednote_research", "2026-08-24T07:00:00+00:00", "completed",
                    ended_at="2026-08-24T07:00:02+00:00")
    await seed_task(_L2_T3, "rednote_research", "2026-08-24T06:00:00+00:00", "failed",
                    ended_at="2026-08-24T06:00:01+00:00")


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_filter_applies_while_hovering(console_stub: SimpleNamespace) -> None:
    """A filter still redraws the list while the mouse sits on a row."""
    await _l2_trio_seed()
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")
        await _wait_for(mgr, tab, _L2_SKELETON_JS, _WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, _L2_SKELETON_GONE_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_ROW_COUNT_JS) == 3
        # Hover the completed row, which keeps its place under the completed filter.
        assert await mgr.evaluate(tab, _js_call(_L2_HOVER_ROW_JS, 1)) == "aaaa0002"
        await mgr.evaluate(tab, _js_call(_L2_CLICK_CHIP_JS, "completed"))
        await _wait_for(mgr, tab, _L2_ROW_COUNT_JS + " === 2", _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_ROW_IDS_JS) == ["aaaa0001", "aaaa0002"]


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_flip_under_parked_cursor_defers_and_repairs_on_pointer_away(
        console_stub: SimpleNamespace) -> None:
    """A change that would move the row under the mouse waits, and lands as soon as the mouse leaves."""
    await seed_task(_L2_T1, "rednote_research", "2026-08-24T06:30:00+00:00")
    await seed_task(_L2_T2, "rednote_research", "2026-08-24T07:00:00+00:00", "completed",
                    ended_at="2026-08-24T07:00:02+00:00")
    await seed_task(_L2_T3, "rednote_research", "2026-08-24T06:00:00+00:00", "failed",
                    ended_at="2026-08-24T06:00:01+00:00")
    stranded = {"running": [_L2_T1], "history": [_L2_T2, _L2_T3]}
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")
        await _wait_for(mgr, tab, _L2_SKELETON_JS, _WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, _L2_SKELETON_GONE_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_GROUPS_JS) == stranded
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 1", "· 2"]
        parked = await _park_pointer(mgr, tab, f"#tt-list .t-row[data-tid=\"{_L2_T2}\"]")
        assert await mgr.evaluate(tab, _js_call(_L2_POINTER_ON_JS, _L2_T2))
        now_iso = (datetime.now(timezone.utc) + timedelta(milliseconds=100)).isoformat()
        await console_stub.finish(_L2_T1, "completed", ended_at=now_iso)
        await _wait_for(mgr, tab, _js_call(_L2_BADGE_COMPLETED_JS, _L2_T1), _WAIT_TIMEOUT_MS)
        # The badge changed, but the rows and the counts have not moved yet.
        assert await mgr.evaluate(tab, _L2_GROUPS_JS) == stranded
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 1", "· 2"]
        await asyncio.sleep(5)  # two poll rounds, no poll moves the held rows
        assert await mgr.evaluate(tab, _L2_GROUPS_JS) == stranded
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 1", "· 2"]
        await _unpark_pointer(mgr, tab, parked)  # fully off the list
        await _wait_for(mgr, tab, _L2_RUN_COUNT_ZERO_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_GROUPS_JS) == {"running": [], "history": [_L2_T2, _L2_T1, _L2_T3]}
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 0", "· 3"]
        # With no running tasks the Running header is hidden.
        assert await mgr.evaluate(tab, _L2_RUN_GROUP_HIDDEN_JS)


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_filter_and_search_apply_under_parked_cursor(console_stub: SimpleNamespace) -> None:
    """A filter and a search take effect right away even while the mouse sits on a row."""
    await _l2_trio_seed()
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")
        await _wait_for(mgr, tab, _L2_SKELETON_JS, _WAIT_TIMEOUT_MS)
        await _wait_for(mgr, tab, _L2_SKELETON_GONE_JS, _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_ROW_COUNT_JS) == 3
        # Sit on the failed row, which the completed filter would take away.
        parked = await _park_pointer(mgr, tab, f"#tt-list .t-row[data-tid=\"{_L2_T3}\"]")
        assert await mgr.evaluate(tab, _js_call(_L2_POINTER_ON_JS, _L2_T3))
        await mgr.evaluate(tab, _js_call(_L2_CLICK_CHIP_JS, "completed"))
        assert await mgr.evaluate(tab, _L2_GROUPS_JS) == {"running": [_L2_T1], "history": [_L2_T2]}
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 1", "· 1"]
        await _unpark_pointer(mgr, tab, parked)
        # Sit on aaaa0002, which the search below takes away.
        parked2 = await _park_pointer(mgr, tab, f"#tt-list .t-row[data-tid=\"{_L2_T2}\"]")
        assert await mgr.evaluate(tab, _js_call(_L2_POINTER_ON_JS, _L2_T2))
        await mgr.type_text(tab, "#tt-q", "aaaa0001")
        await _wait_for(mgr, tab, _L2_ROW_COUNT_JS + " === 1", _WAIT_TIMEOUT_MS)
        assert await mgr.evaluate(tab, _L2_GROUPS_JS) == {"running": [_L2_T1], "history": []}
        assert await mgr.evaluate(tab, _L2_GROUP_COUNTS_JS) == ["· 1", "· 0"]
        await _unpark_pointer(mgr, tab, parked2)


_L1_STAMP_AHEAD_MIN = 10  # minutes the fake server time runs ahead of the browser clock
_L1_IDLE_AGE_S = 60  # seconds the idle tool was last busy before the fake server time
_L1_RUN_STARTED_S = 300  # seconds the running tool's run started before the fake server time
_L1_DUR_TOL_S = 10


def _l1_skew_tools(server_now: datetime) -> dict:
    """A tool list stamped with the given server time, which still runs ahead of the browser clock.

    The running tool's run started before that time.
    """
    gen = server_now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    gen_local = server_now.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    run_since = (server_now - timedelta(seconds=_L1_RUN_STARTED_S)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    idle = server_now - timedelta(seconds=_L1_IDLE_AGE_S)
    return {
        "tools": [
            {"name": "skew_run", "description": "A running stub tool", "running": 1,
             "completed": 0, "failed": 0, "cancelled": 0,
             "running_since": run_since, "last_activity": gen, "last_activity_local": gen_local},
            {"name": "skew_idle", "description": "An idle stub tool", "running": 0,
             "completed": 5, "failed": 0, "cancelled": 0,
             "last_activity": idle.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
             "last_activity_local": idle.astimezone().strftime("%Y-%m-%d %H:%M:%S")},
        ],
        "generated_at": gen, "generated_at_local": gen_local,
    }


def _skew_changes(server_now: datetime) -> dict:
    """A changes reply stamped with the given server time, carrying no tool stats."""
    gen = server_now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return {"fingerprint": "fp0", "generated_at": gen,
            "generated_at_local": server_now.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {"tools": {}}, "running": [], "started": [], "finished": []}


_L1_ROWS_READY_JS = "() => document.querySelectorAll('#l1-list .l1-row').length === 2"
_L1_RUN_DUR_JS = """
() => document.querySelector('#l1-list .l1-row[data-name="skew_run"] .c-last').textContent
"""
_L1_IDLE_REL_JS = """
() => document.querySelector('#l1-list .l1-row[data-name="skew_idle"] .c-last').textContent
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l1_running_duration_tracks_skewed_server_stamp(console_stub: SimpleNamespace) -> None:
    """With the server clock 10 minutes ahead, the tools screen shows the run's elapsed time, not the wall clock."""
    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    server_now = t0 + timedelta(minutes=_L1_STAMP_AHEAD_MIN)
    # The changes polls re-stamp the server time, so they must stay skewed too.
    console_stub.pin_body(r"/api/tools$", _l1_skew_tools(server_now))
    console_stub.pin_body(r"/api/changes$", _skew_changes(server_now))
    base = console_stub.base_url

    def _expected_elapsed() -> float:
        return _L1_RUN_STARTED_S + (datetime.now(timezone.utc) - t0).total_seconds()

    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools")
        await _wait_for(mgr, tab, _L1_ROWS_READY_JS, _WAIT_TIMEOUT_MS)
        await asyncio.sleep(3)  # one poll round, later polls must keep the elapsed time
        rendered = _fmt_dur_to_seconds(await mgr.evaluate(tab, _L1_RUN_DUR_JS))
        assert abs(rendered - _expected_elapsed()) <= _L1_DUR_TOL_S
        assert rendered >= _L1_RUN_STARTED_S - _L1_DUR_TOL_S  # the browser clock alone would show 0
        assert await mgr.evaluate(tab, _L1_IDLE_REL_JS) == "1m ago"


_L2_SKEW_TASK_ID = "bbbb0001-0000-0000-0000-000000000001"
_L2_SKEW_AHEAD_MIN = 10  # minutes the fake server time runs ahead of the browser clock
_L2_SKEW_TASK_STARTED_S = 300  # seconds the running task started before the fake server time
_L2_SKEW_TOL_S = 10


def _fmt_dur_to_seconds(text: str) -> int:
    """Turn a duration like 5m 3s back into seconds.

    Args:
        text: The duration text.

    Returns:
        How many seconds it means.

    Raises:
        ValueError: If the text has another shape.
    """
    m = re.fullmatch(r"(\d+)m (\d+)s", text)
    if m is None:
        raise ValueError("unexpected duration text: " + text)
    return int(m.group(1)) * 60 + int(m.group(2))


def _l2_skew_tools(server_now: datetime) -> dict:
    """A tool list stamped with the given server time, showing one running rednote_research run."""
    gen = server_now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    gen_local = server_now.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "tools": [
            {"name": "rednote_research", "description": "Skew stub tool", "running": 1,
             "completed": 0, "failed": 0, "cancelled": 0,
             "running_since": (server_now - timedelta(seconds=_L2_SKEW_TASK_STARTED_S))
             .strftime("%Y-%m-%dT%H:%M:%S+00:00"),
             "last_activity": gen, "last_activity_local": gen_local},
        ],
        "generated_at": gen, "generated_at_local": gen_local,
    }


_L2_SKEW_C_DUR_JS = """
() => {
  const n = document.querySelector('#tt-list .t-row .c-dur');
  return n === null ? '' : n.textContent;
}
"""
_L2_SKEW_C_DUR_READY_JS = """
() => {
  const n = document.querySelector('#tt-list .t-row .c-dur');
  return n !== null && n.textContent !== '';
}
"""
_D_SKEW_ELAPSED_JS = """
() => {
  const n = document.querySelector('#d-elapsed');
  return n === null ? '' : n.textContent;
}
"""
_D_SKEW_ELAPSED_READY_JS = """
() => {
  const n = document.querySelector('#d-elapsed');
  return n !== null && n.textContent !== '';
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l2_and_detail_durations_track_skewed_server_stamp(console_stub: SimpleNamespace) -> None:
    """With the server clock 10 minutes ahead, the running times on the list and on the task page follow server time."""
    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    server_now = t0 + timedelta(minutes=_L2_SKEW_AHEAD_MIN)
    # The task starts before the skewed server time, which is still ahead of the browser clock.
    started_iso = (server_now - timedelta(seconds=_L2_SKEW_TASK_STARTED_S)).isoformat()
    await seed_task(_L2_SKEW_TASK_ID, "rednote_research", started_iso)
    # The changes polls re-stamp the server time, so they must stay skewed too.
    console_stub.pin_body(r"/api/tools$", _l2_skew_tools(server_now))
    console_stub.pin_body(r"/api/changes$", _skew_changes(server_now))
    base = console_stub.base_url

    def _expected_elapsed() -> float:
        return _L2_SKEW_TASK_STARTED_S + (datetime.now(timezone.utc) - t0).total_seconds()

    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")
        await _wait_for(mgr, tab, _L2_SKEW_C_DUR_READY_JS, _WAIT_TIMEOUT_MS)
        rendered = _fmt_dur_to_seconds(await mgr.evaluate(tab, _L2_SKEW_C_DUR_JS))
        assert abs(rendered - _expected_elapsed()) <= _L2_SKEW_TOL_S
        assert rendered >= _L2_SKEW_TASK_STARTED_S - _L2_SKEW_TOL_S  # the browser clock alone would show 0
        await mgr.navigate(tab, base + f"/tools/rednote_research/{_L2_SKEW_TASK_ID}")
        await _wait_for(mgr, tab, _D_SKEW_ELAPSED_READY_JS, _WAIT_TIMEOUT_MS)
        elapsed = _fmt_dur_to_seconds(await mgr.evaluate(tab, _D_SKEW_ELAPSED_JS))
        assert abs(elapsed - _expected_elapsed()) <= _L2_SKEW_TOL_S


_L1_FLIP_TOOLS = [
    {"name": "aaa_idle", "description": "First stub tool", "running": 0, "completed": 3, "failed": 0, "cancelled": 0,
     "last_activity": None, "last_activity_local": None},
    {"name": "mmm_idle", "description": "Middle stub tool", "running": 0, "completed": 0, "failed": 0, "cancelled": 0,
     "last_activity": None, "last_activity_local": None},
    {"name": "zzz_idle", "description": "Last stub tool", "running": 0, "completed": 1, "failed": 0, "cancelled": 0,
     "last_activity": None, "last_activity_local": None},
]
_L1_FLIP_ORDER = "aaa_idle,mmm_idle,zzz_idle"
_L1_ZZZ_TOP_ORDER = "zzz_idle,aaa_idle,mmm_idle"

_L1_ROW_ORDER_JS = """
() => [...document.querySelectorAll('#l1-list .l1-row')].map((r) => r.dataset.name).join(',')
"""
_L1_ZZZ_LIVE_JS = """
() => {
  const run = document.querySelector('#l1-list .l1-row[data-name="zzz_idle"] .c-run');
  return run !== null && run.classList.contains("live") && run.textContent === "1";
}
"""
_L1_POINTER_ON_JS = """
(name) => {
  const r = document.querySelector('#l1-list .l1-row[data-name="' + name + '"]');
  const b = r.getBoundingClientRect();
  const hit = document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2);
  return hit !== null && hit.closest(".l1-row") === r;
}
"""


def _l1_zzz_changes(stamp: str, stamp_local: str) -> dict:
    """A changes reply that reports the last tool as newly running, and the others idle."""
    return {"fingerprint": "fp1", "generated_at": stamp, "generated_at_local": stamp_local,
            "stats": {"tools": {"zzz_idle": {"running": 1, "completed": 1, "failed": 0, "cancelled": 0,
                                             "running_since": stamp, "last_activity": stamp,
                                             "last_activity_local": stamp_local}}},
            "running": [], "started": [], "finished": []}


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_l1_flip_under_parked_cursor_defers_and_repairs_on_pointer_away(
        console_stub: SimpleNamespace) -> None:
    """A poll that would move the row under the mouse waits, and lands as soon as the mouse leaves."""
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    stamp_local = now.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    tools_body = {"tools": list(_L1_FLIP_TOOLS), "generated_at": stamp, "generated_at_local": stamp_local}
    empty_changes = {"fingerprint": "fp0", "generated_at": stamp, "generated_at_local": stamp_local,
                     "stats": {"tools": {}}, "running": [], "started": [], "finished": []}
    flipped = {"v": False}

    def _changes_body(_scope: dict) -> dict:
        return _l1_zzz_changes(stamp, stamp_local) if flipped["v"] else empty_changes

    console_stub.pin_body(r"/api/tools$", tools_body)
    console_stub.pin_dynamic(r"/api/changes$", _changes_body)
    base = console_stub.base_url
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools")
        await _wait_for(mgr, tab, _L1_ROW_ORDER_JS + " === '" + _L1_FLIP_ORDER + "'", _WAIT_TIMEOUT_MS)
        parked = await _park_pointer(mgr, tab, '#l1-list .l1-row[data-name="mmm_idle"]')
        assert await mgr.evaluate(tab, _js_call(_L1_POINTER_ON_JS, "mmm_idle"))
        flipped["v"] = True
        await _wait_for(mgr, tab, _L1_ZZZ_LIVE_JS, _WAIT_TIMEOUT_MS)
        # The new numbers are in, but the rows have not moved.
        assert await mgr.evaluate(tab, _L1_ROW_ORDER_JS) == _L1_FLIP_ORDER
        await asyncio.sleep(5)  # two poll rounds, no poll moves the held rows
        assert await mgr.evaluate(tab, _L1_ROW_ORDER_JS) == _L1_FLIP_ORDER
        await _unpark_pointer(mgr, tab, parked)  # fully off the list
        await _wait_for(mgr, tab, _L1_ROW_ORDER_JS + " === '" + _L1_ZZZ_TOP_ORDER + "'", _WAIT_TIMEOUT_MS)


_SIDEBAR_STATUS_JS = """
() => ({
  endpoint: document.getElementById("sb-endpoint").textContent,
  text: document.getElementById("sidebar-status").textContent,
  title: document.getElementById("sidebar-status").title,
})
"""
_BROWSER_GONE_JS = "() => document.querySelector('#br-gone .t2').textContent"


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_sidebar_and_browser_screen_show_no_ports(console_stub: SimpleNamespace) -> None:
    """The sidebar shows the MCP endpoint from the server, and no screen names a fixed port."""
    base = console_stub.base_url
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        dav_base = (await c.get("/api/health")).json()["dav_base_url"]
    endpoint = f"v1 · MCP {dav_base}/mcp"
    async with e2e_page(_DESKTOP_VIEWPORT) as (mgr, tab):
        await mgr.navigate(tab, base + "/browser")
        await mgr.wait_for_selector(tab, "#br-gone", timeout=_WAIT_TIMEOUT_MS)
        await _wait_for(
            mgr, tab, "() => document.getElementById('sb-endpoint').textContent !== 'v1'", _WAIT_TIMEOUT_MS
        )
        status = await mgr.evaluate(tab, _SIDEBAR_STATUS_JS)
        gone = await mgr.evaluate(tab, _BROWSER_GONE_JS)
    assert status["endpoint"] == endpoint
    assert status["title"].endswith(" · " + endpoint)
    copy = " ".join([status["text"], status["title"]]).replace(dav_base, "") + " " + gone
    assert ":8315" not in copy
    assert "6080" not in copy
    assert gone == "The VNC stream was closed or is unreachable."


@pytest.mark.live
@pytest.mark.timeout(30)
async def test_browser_screen_and_vnc_status(app_http: SimpleNamespace) -> None:
    """The browser screen loads, the screen sharing page answers, and the status says it is reachable."""
    require_novnc()
    async with httpx.AsyncClient(base_url=app_http.base_url, timeout=10) as c:
        assert (await c.get("/browser")).status_code == 200
        assert "/assets/js/screens/browser.js" in (await c.get("/browser")).text
        assert (await c.get("/assets/js/screens/browser.js")).status_code == 200
        assert (await c.get("/api/vnc/status")).json()["reachable"] is True
        assert (await c.get("/vnc/vnc.html")).status_code == 200


@pytest.mark.live
@pytest.mark.timeout(30)
async def test_websockify_ws_reaches_vnc_proxy(app_http: SimpleNamespace) -> None:
    """A websocket connect on the app port reaches the screen sharing server."""
    require_novnc()
    async with connect(f"ws://127.0.0.1:{app_http.port}/websockify", open_timeout=5) as ws:
        banner = await asyncio.wait_for(ws.recv(), timeout=10)
    assert banner.startswith(b"RFB ")



def _asset_text(client: TestClient, path: str) -> str:
    """Read a frontend asset from the server."""
    response = client.get("/assets/" + path)
    response.raise_for_status()
    return response.text


def test_fmt_media_clock_hours(tmp_path: Path, frontend_assets: TestClient) -> None:
    """Media times include hours when needed."""
    mod = tmp_path / "audio-card.mjs"

    mod.write_text(_asset_text(frontend_assets, "js/audio-card.js"), encoding="utf-8")
    code = ('import {fmtMediaClock as f} from "' + mod.as_uri() + '";\n'
            "console.log([f(187), f(59), f(3599), f(3600), f(3727), f(7261)].join('|'));")
    out = subprocess.run(["node", "--input-type=module", "-e", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().split("|") == ["3:07", "0:59", "59:59", "1:00:00", "1:02:07", "2:01:01"]


@pytest.mark.timeout(30)
async def test_console_stub_serves_the_real_frontend(console_stub: SimpleNamespace) -> None:
    """The server serves pages, assets, and redirects."""
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as c:
        r = await c.get("/tools")
        assert r.status_code == 200
        assert r.text == (await c.get("/assets/index.html")).text
        r = await c.get("/assets/js/api.js")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/javascript")
        r = await c.get("/some/unknown/spa/route", follow_redirects=False)
        assert r.status_code == 200
        assert r.text.startswith("<!doctype html>")
        r = await c.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/tools"

_STUB_SMOKE_TIDS = ("stub-smoke-1", "stub-smoke-2", "stub-smoke-3")
_STUB_SMOKE_DESKTOP = dict(_DESKTOP_VIEWPORT)
_STUB_SMOKE_TIMEOUT_MS = 10000
_STUB_SMOKE_POLL_S = 0.2
_STUB_SMOKE_ALL_DONE_JS = """
() => {
  const badges = document.querySelectorAll('#tt-list .t-row .badge');
  return badges.length === 3 && [...badges].every((b) => b.textContent === 'completed');
}
"""


@pytest.mark.live
@pytest.mark.timeout(120)
@skip_if_no_chrome
async def test_console_stub_backend_drives_the_real_frontend(console_stub: SimpleNamespace) -> None:
    """The research screen shows task completion without a reload."""
    await console_stub.seed_task(_STUB_SMOKE_TIDS[0], "rednote_research", "2026-08-23T01:00:00+00:00", "completed",
                                 ended_at="2026-08-23T01:00:05+00:00", params={"query": "first"})
    await console_stub.seed_task(_STUB_SMOKE_TIDS[1], "rednote_research", "2026-08-23T02:00:00+00:00", "completed",
                                 ended_at="2026-08-23T02:00:05+00:00", params={"query": "second"})
    await console_stub.seed_task(_STUB_SMOKE_TIDS[2], "rednote_research", "2026-08-23T03:00:00+00:00",
                                 params={"query": "live"})
    base = console_stub.base_url
    async with e2e_page(_STUB_SMOKE_DESKTOP) as (mgr, tab):
        await mgr.navigate(tab, base + "/tools/rednote_research")

        async def wait_true(expression: str) -> None:
            deadline = time.monotonic() + _STUB_SMOKE_TIMEOUT_MS / _MS_PER_S
            while True:
                if await mgr.evaluate(tab, expression):
                    return
                if time.monotonic() >= deadline:
                    raise AssertionError("timed out waiting for: " + expression[:120])
                await asyncio.sleep(_STUB_SMOKE_POLL_S)

        await wait_true("() => document.querySelectorAll('#tt-list .t-row').length === 3")
        badges = await mgr.evaluate(
            tab, "() => [...document.querySelectorAll('#tt-list .t-row .badge')].map((b) => b.textContent)")
        assert badges.count("running") == 1
        assert badges.count("completed") == 2
        await console_stub.finish(_STUB_SMOKE_TIDS[2], "completed", ended_at="2030-01-01T00:00:00+00:00")
        await wait_true(_STUB_SMOKE_ALL_DONE_JS)
        assert await mgr.evaluate(tab, "() => document.querySelector('#g-run').style.display") == "none"


async def _seed_conv_fold_task(console_stub: SimpleNamespace, tid: str, status: str, ended_at: str) -> None:
    await console_stub.seed_task(tid, "planner", "2026-08-23T01:00:00+00:00", status, ended_at=ended_at)
    await store_module.agent_store.insert(tid, "planner")
    nd = "\n".join([
        json.dumps({"type": "system", "attempt": 1, "text": "You drive a real browser to carry out a goal across open tabs. " * 20, "ts": "t"}),
        json.dumps({"type": "user", "attempt": 1, "text": "Summarize the page.", "ts": "t"}),
        json.dumps({"type": "assistant", "attempt": 1, "text": "Here is the summary of the results. " * 40, "ts": "t"}),
    ]) + "\n"
    write_run_file(console_stub.data_dir, tid, "conversation.jsonl", nd)


@pytest.mark.live
@pytest.mark.timeout(180)
@skip_if_no_chrome
async def test_conv_fold_live_geometry(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """A long answer expands by keyboard and stays collapsible after a resize."""
    page = _parameter_page
    await _seed_conv_fold_task(console_stub, "fold-accessible", "completed", E2E_ENDED_AT)
    await page.goto(console_stub.base_url + "/task/fold-accessible")
    await page.get_by_role("button", name="Conversation", exact=True).click()
    card = page.locator(".conv-assist")
    disclosure = card.get_by_role("button", name=re.compile("Assistant"))
    await expect(disclosure).to_have_attribute("aria-expanded", "false")
    await expect(disclosure).to_have_text(re.compile(r"[1-9][0-9]* lines? hidden"))
    initial = await card.bounding_box()
    await disclosure.focus()
    await page.keyboard.press("Enter")
    await expect(disclosure).to_have_attribute("aria-expanded", "true")
    await page.wait_for_function("""height => document.querySelector('.conv-assist').getBoundingClientRect().height > height""", arg=initial["height"])
    await page.keyboard.press("Enter")
    await expect(disclosure).to_have_attribute("aria-expanded", "false")
    await page.set_viewport_size({"width": 390, "height": _PARAMETER_PHONE_HEIGHT})
    await expect(disclosure).to_have_attribute("aria-expanded", "false")
    await expect(disclosure).to_have_text(re.compile(r"[1-9][0-9]* lines? hidden"))
    await expect(page.get_by_text("Summarize the page.", exact=True)).to_be_visible()


@pytest.mark.live
@pytest.mark.timeout(180)
@skip_if_no_chrome
async def test_conv_fold_live_mobile_label_row_geometry(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """The phone disclosure label and answer stay within the card without overlap."""
    page = _parameter_page
    await page.set_viewport_size({"width": 390, "height": _PARAMETER_PHONE_HEIGHT})
    await _seed_conv_fold_task(console_stub, "fold-phone", "completed", E2E_ENDED_AT)
    await page.goto(console_stub.base_url + "/task/fold-phone")
    await page.get_by_role("button", name="Conversation", exact=True).click()
    card = page.locator(".conv-assist")
    disclosure = card.get_by_role("button", name=re.compile("Assistant"))
    await expect(disclosure).to_have_attribute("aria-expanded", "false")
    label_box = await disclosure.bounding_box()
    body_box = await card.locator(".conv-assist-body").bounding_box()
    assert label_box["x"] >= 0
    assert label_box["x"] + label_box["width"] <= 390
    assert label_box["y"] + label_box["height"] <= body_box["y"] + 1
    assert body_box["y"] - label_box["y"] - label_box["height"] < label_box["height"]


@pytest.mark.live
@pytest.mark.timeout(180)
@skip_if_no_chrome
async def test_conv_fold_live_terminal_settles_the_open_card(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """Completion collapses the answer without duplicating it or losing its text."""
    page = _parameter_page
    tid = "fold-live-completion"
    answer = "Working through the steps. " * 40
    await console_stub.seed_task(tid, "planner", E2E_STARTED_AT)
    await store_module.agent_store.insert(tid, "planner")
    content = json.dumps({"type": "content", "attempt": 1, "text": answer, "ts": "t"}) + "\n"
    await asyncio.to_thread(write_run_file, console_stub.data_dir, tid, "conversation.jsonl", content)
    await page.goto(console_stub.base_url + "/task/" + tid)
    await page.get_by_role("button", name="Conversation", exact=True).click()
    await expect(page.locator(".conv-assist")).to_have_count(1)
    await expect(page.get_by_text(answer.strip(), exact=True)).to_be_visible()
    await console_stub.finish(tid, "failed", ended_at="2030-01-01T00:00:00+00:00")
    disclosure = page.locator(".conv-assist").get_by_role("button", name=re.compile("Assistant"))
    await expect(disclosure).to_have_attribute("aria-expanded", "false", timeout=_STUB_SMOKE_TIMEOUT_MS)
    await expect(page.locator(".conv-assist")).to_have_count(1)
    assert await page.locator(".conv-assist-body").text_content() == answer.strip()
    await disclosure.click()
    await expect(disclosure).to_have_attribute("aria-expanded", "true")
    await expect(page.get_by_text(answer.strip(), exact=True)).to_be_visible()


async def _served_surface(stub: SimpleNamespace, page: Page, width: int) -> None:
    """Open the served page for a component test."""
    await page.set_viewport_size({"width": width, "height": _PARAMETER_PHONE_HEIGHT})
    await page.goto(stub.base_url + "/tools")
    await page.locator("#view").wait_for()
    await page.evaluate("""() => {
      const surface = document.createElement('main');
      surface.id = 'surface';
      surface.style.cssText = 'width:100%;min-width:0;padding:16px;overflow:auto';
      document.body.replaceChildren(surface);
    }""")
    await page.wait_for_function("() => document.fonts.status === 'loaded'")


async def _served_conversation(stub: SimpleNamespace, page: Page, subtasks: list[dict]) -> None:
    """Open a live conversation on the served page."""
    tid = "conversation-contract"
    await stub.seed_task(tid, "planner", E2E_STARTED_AT)
    await _served_surface(stub, page, _DESKTOP_VIEWPORT["width"])
    await page.evaluate("""async detail => {
      const { mountConversation } = await import('/assets/js/screens/conversation.js');
      window.conversation = mountConversation(document.getElementById('surface'), detail);
    }""", {"task": {"task_id": tid, "status": "running"}, "subtasks": subtasks})


async def _conversation_events(page: Page, events: list[dict]) -> None:
    """Deliver conversation events to the reader."""
    await page.evaluate("text => window.conversation.ingest(text)", "\n".join(json.dumps(event) for event in events))


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_conversation_keeps_live_text_and_final_answer(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """Live and final answers keep their text and inner blank lines."""
    page = _parameter_page
    await _served_conversation(console_stub, page, [])
    await _conversation_events(page, [{"type": "content", "attempt": 1, "text": "\n\n**Hello**\n\nworld\n\n"}])
    answer = page.locator(".conv-assist")
    assert await answer.locator(".conv-assist-body").inner_text() == "**Hello**\n\nworld"
    async with page.expect_response("**/api/render/message") as rendered:
        await _conversation_events(page, [{"type": "assistant", "attempt": 1, "text": "\n\nFinal answer.\n\nSecond paragraph.\n\n"}])
    await (await rendered.value).finished()
    await expect(answer.locator(".conv-assist-body")).to_have_text("Final answer.\n\nSecond paragraph.")
    await expect(answer).to_have_count(1)


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("result_task,expected", [
    ("child-cached", ["/task/child-cached", "/task/child-a"]),
    (None, [None, "/task/child-a"]),
    ("child-a", ["/task/child-a", None]),
])
async def test_served_conversation_links_each_call_to_its_result(
    console_stub: SimpleNamespace, _parameter_page: Page, result_task: str | None, expected: list[str | None]
) -> None:
    """Calls link to their own tasks even when a result uses a saved task."""
    page = _parameter_page
    tool = "search_evidence"
    await _served_conversation(console_stub, page, [{
        "task_id": "child-a", "parent_task_id": "conversation-contract", "tool": tool, "status": "running",
    }])
    await _conversation_events(page, [
        {"type": "tool_call", "attempt": 1, "call_id": "first", "tool_name": tool, "ts": E2E_STARTED_AT},
        {"type": "tool_call", "attempt": 1, "call_id": "second", "tool_name": tool, "ts": E2E_STARTED_AT},
        {"type": "tool_result", "attempt": 1, "call_id": "first", "tool_name": tool,
         "task_id": result_task, "ts": E2E_ENDED_AT},
    ])
    calls = page.locator("a").filter(has_text=tool)
    await expect(calls).to_have_count(2)
    assert await calls.nth(0).get_attribute("href") == expected[0]
    assert await calls.nth(1).get_attribute("href") == expected[1]


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_conversation_late_render_cannot_replace_newer_answer(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """An older reply cannot replace the latest answer."""
    page = _parameter_page
    await _served_conversation(console_stub, page, [])
    release = asyncio.Event()
    received = asyncio.Event()
    delivered = asyncio.Event()

    async def delay_old_reply(route: PlaywrightRoute) -> None:
        response = await route.fetch()
        if route.request.post_data == "Older answer.":
            received.set()
            await release.wait()
            await route.fulfill(response=response)
            delivered.set()
        else:
            await route.fulfill(response=response)

    await page.route("**/api/render/message", delay_old_reply)
    await _conversation_events(page, [{"type": "assistant", "attempt": 1, "text": "Older answer."}])
    await asyncio.wait_for(received.wait(), timeout=10)
    await _conversation_events(page, [{"type": "assistant", "attempt": 1, "text": "Latest answer."}])
    answer = page.locator(".conv-assist")
    await expect(answer.locator(".conv-assist-body")).to_have_text("Latest answer.")
    release.set()
    await asyncio.wait_for(delivered.wait(), timeout=10)
    await page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    await expect(answer.locator(".conv-assist-body")).to_have_text("Latest answer.")
    await expect(answer).to_have_count(1)


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("width", [390, _WIDE_DESKTOP_VIEWPORT["width"]])
@pytest.mark.parametrize("action", ["none", "remove", "copy"])
async def test_served_resource_card_actions(
    console_stub: SimpleNamespace, _parameter_page: Page, width: int, action: str
) -> None:
    """File links remain usable beside the requested action at each screen size."""
    page = _parameter_page
    await _served_surface(console_stub, page, width)
    await page.evaluate("""async (action) => {
      const { renderResourceCard } = await import('/assets/js/resource-card.js');
      window.actionCalls = [];
      const opts = {};
      if (action === 'remove') opts.onRemove = () => actionCalls.push('remove');
      if (action === 'copy') opts.onCopy = () => actionCalls.push('copy');
      renderResourceCard(document.getElementById('surface'), {
        path: 'upload/a long file name.txt', name: 'a long file name.txt', size: 1024, kind: 'text',
      }, opts);
    }""", action)
    link = page.get_by_role("link", name="Open a long file name.txt in a new tab")
    await expect(link).to_have_attribute("href", "/view/upload/a%20long%20file%20name.txt")
    await expect(link).to_have_attribute("target", "_blank")
    await expect(page.get_by_text("upload · 1 KB", exact=False)).to_be_visible()
    expected = {"none": None, "remove": "Remove", "copy": "Copy path"}[action]
    await expect(page.get_by_role("button")).to_have_count(0 if expected is None else 1)
    if expected is not None:
        button = page.get_by_role("button", name=expected, exact=True)
        await button.click()
        assert await page.evaluate("actionCalls") == ["copy" if action == "copy" else "remove"]
        link_box, button_box = await link.bounding_box(), await button.bounding_box()
        assert link_box["x"] + link_box["width"] <= button_box["x"] + 1
        assert button_box["x"] + button_box["width"] <= width
        if width == 390:
            assert button_box["width"] >= 44
            assert button_box["height"] >= 44
    async with page.expect_popup() as opened:
        await link.focus()
        await page.keyboard.press("Space")
    popup = await opened.value
    assert popup.url.endswith("/view/upload/a%20long%20file%20name.txt")
    await popup.close()


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_resource_card_resolves_paths_and_missing_files(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """Bare paths gain metadata and missing files lose their open action."""
    page = _parameter_page
    console_stub.pin_body(r"/api/resources$", {"entries": [
        {"path": "upload/found.txt", "name": "found.txt", "size": 1024, "kind": "text"},
    ]})
    await _served_surface(console_stub, page, 390)
    await page.evaluate("""async () => {
      const { renderResourceCard } = await import('/assets/js/resource-card.js');
      const host = document.getElementById('surface');
      renderResourceCard(host, 'upload/found.txt');
      renderResourceCard(host, 'upload/missing.txt', { onRemove() { window.removedMissing = true; } });
    }""")
    await expect(page.get_by_text("upload · 1 KB", exact=True)).to_be_visible()
    await expect(page.get_by_text("Removed by 30-day retention · no longer exists")).to_be_visible()
    await expect(page.get_by_role("link")).to_have_count(1)
    await expect(page.get_by_role("link")).to_have_accessible_name("Open found.txt in a new tab")
    await page.get_by_role("button", name="Remove", exact=True).click()
    assert await page.evaluate("window.removedMissing") is True


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("gone", [False, True])
async def test_served_resource_card_pending_thumbnail_settles(
    console_stub: SimpleNamespace, _parameter_page: Page, gone: bool
) -> None:
    """A pending file becomes an image or an unavailable file after the probe."""
    page = _parameter_page
    await page.route("**/api/resources/thumb?*", lambda route: route.fulfill(
        status=404 if gone else 200, content_type="image/png", body=b"" if gone else E2E_PNG
    ))
    await _served_surface(console_stub, page, _WIDE_DESKTOP_VIEWPORT["width"])
    await page.evaluate("""async () => {
      const { renderResourceCard } = await import('/assets/js/resource-card.js');
      window.cardFacts = [];
      renderResourceCard(document.getElementById('surface'), {
        path: 'upload/pending.png', name: 'pending.png', thumb: { status: 'pending' },
      }, { onFact: (fact) => cardFacts.push(fact ? fact.thumb.status : null) });
    }""")
    if gone:
        await expect(page.get_by_text("Removed by 30-day retention · no longer exists")).to_be_visible()
        await expect(page.get_by_role("link")).to_have_count(0)
        assert await page.evaluate("cardFacts") == ["pending", None]
    else:
        await expect(page.locator("img")).to_be_visible()
        await page.wait_for_function("() => document.querySelector('img').naturalWidth > 0")
        assert await page.evaluate("cardFacts") == ["pending", "done"]


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_gallery_orders_entries_and_supports_keyboard_selection(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """Folders sort by name before recent files and keyboard activation selects a file."""
    page = _parameter_page
    await _served_surface(console_stub, page, _WIDE_DESKTOP_VIEWPORT["width"])
    await page.evaluate("""async () => {
      const { renderGallery } = await import('/assets/js/gallery.js');
      const host = document.getElementById('surface');
      host.addEventListener('gallery:select', (ev) => { window.selectedFile = ev.detail.path; });
      host.addEventListener('gallery:open', (ev) => { window.openedFile = ev.detail.path; });
      renderGallery([
        { path: 'old.txt', name: 'old.txt', type: 'file', mtime: 1, size: 10 },
        { path: 'Z folder', name: 'Z folder', type: 'dir', file_count: 0 },
        { path: 'new.txt', name: 'new.txt', type: 'file', mtime: 2, size: 20 },
        { path: 'A folder', name: 'A folder', type: 'dir', file_count: 1 },
      ], [], host);
    }""")
    titles = await page.locator("[role=button][title]").evaluate_all("els => els.map(el => el.title)")
    assert titles == ["A folder", "Z folder", "new.txt", "old.txt"]
    selected = page.locator('[role="button"][title="new.txt"]')
    await selected.focus()
    await page.keyboard.press("Enter")
    assert await page.evaluate("selectedFile") == "new.txt"
    await selected.get_by_role("button", name="Open in full screen — this console").click()
    assert await page.evaluate("openedFile") == "new.txt"
    await expect(selected.get_by_role("link", name="Download")).to_have_attribute("href", "/dav/dl?path=new.txt")


_WRAPPED_FRAGMENTS = {
    "plain": '<pre class="plain-pre">{text}</pre>',
    "code": '<pre class="code-block"><span class="cl">{text}</span></pre>',
    "markdown_code": '<article class="md-doc"><pre><code>{text}</code></pre></article>',
    "markdown_table": '<article class="md-doc"><div class="md-table-scroll"><table><tbody><tr><td>{text}</td></tr></tbody></table></div></article>',
    "csv": '<table class="csv-table"><tbody><tr><td>{text}</td></tr></tbody></table>',
}


def _result_descriptor(fragment: str, mode: str = "plain", kind: str = "text") -> dict:
    """Describe one ready result body for the frontend."""
    return {"size": _FRONTEND_FILE_SIZE, "mode": mode, "kind": kind, "detected_kind": kind, "fragment": fragment}


async def _mount_served_result(stub: SimpleNamespace, page: Page, descriptor: dict, width: int) -> None:
    """Mount a result against the server feed."""
    stub.pin_body(r"/api/tasks/served-contract/result/feed$", {"body": descriptor, "files": []})
    await _served_surface(stub, page, width)
    await page.evaluate("""async () => {
      const { mountResult } = await import('/assets/js/screens/result.js');
      window.resultHandle = mountResult(document.getElementById('surface'), {
        id: 'served-contract', task: { status: 'completed' },
      });
    }""")
    await page.get_by_role("button", name="Preview", exact=True).wait_for()


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("width", [390, _WIDE_DESKTOP_VIEWPORT["width"]])
@pytest.mark.parametrize("format_name", list(_WRAPPED_FRAGMENTS))
async def test_served_result_long_content_remains_accessible(
    console_stub: SimpleNamespace, _parameter_page: Page, width: int, format_name: str
) -> None:
    """Long previews remain readable inside the screen without losing content."""
    page = _parameter_page
    text = "LongUnbrokenValue" * 40
    mode = {"plain": "plain", "code": "code", "csv": "table"}.get(format_name, "doc")
    descriptor = _result_descriptor(_WRAPPED_FRAGMENTS[format_name].format(text=text), mode)
    await _mount_served_result(console_stub, page, descriptor, width)
    content = page.get_by_text(text, exact=True)
    await expect(content).to_be_visible()
    geometry = await content.evaluate("""el => {
      const range = document.createRange(); range.selectNodeContents(el);
      const rects = [...range.getClientRects()];
      return {
        lines: new Set(rects.map(r => Math.round(r.top))).size,
        right: Math.max(...rects.map(r => r.right)),
        scroll: document.documentElement.scrollWidth,
        width: innerWidth,
      };
    }""")
    if format_name == "markdown_table" and width == 390:
        scroll = page.locator(".md-table-scroll")
        assert await scroll.evaluate("el => el.scrollWidth > el.clientWidth")
        await scroll.evaluate("el => { el.scrollLeft = el.scrollWidth; }")
        assert await scroll.evaluate("el => el.scrollLeft") > 0
    else:
        assert geometry["lines"] > 1
        assert geometry["right"] <= width
    assert geometry["scroll"] <= geometry["width"]
    assert await content.text_content() == text


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("width", [390, _WIDE_DESKTOP_VIEWPORT["width"]])
async def test_served_result_kind_change_keeps_fullscreen_selection(
    console_stub: SimpleNamespace, _parameter_page: Page, width: int
) -> None:
    """Changing the preview kind updates the result and its fullscreen link."""
    page = _parameter_page
    next_body = _result_descriptor('<article class="md-doc"><strong>Formatted answer</strong></article>', "doc", "markdown")
    console_stub.pin_body(r"/api/tasks/served-contract/result/view$", next_body)
    await _mount_served_result(console_stub, page, _result_descriptor('<pre class="plain-pre">Raw answer</pre>'), width)
    if width == 390:
        control = page.get_by_role("combobox", name="Preview kind", exact=True)
        await expect(control).to_be_visible()
        await control.select_option("markdown")
    else:
        await page.get_by_title("Preview kind", exact=True).click()
        await page.get_by_text("Auto-detect", exact=True).last.wait_for()
        await page.get_by_text("markdown", exact=True).last.click()
    await expect(page.get_by_text("Formatted answer", exact=True)).to_be_visible()
    await expect(page.get_by_role("link", name="Open full-screen", exact=True)).to_have_attribute(
        "href", "/view/result/served-contract?kind=markdown"
    )
    await expect(page.get_by_role("link", name="Download", exact=True)).to_have_attribute(
        "href", "/api/tasks/served-contract/result/download"
    )


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_result_declared_kind_is_fixed(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """A tool declared preview kind is a label and cannot be overridden."""
    descriptor = {**_result_descriptor('<pre class="plain-pre">Fixed answer</pre>'), "declared_kind": "text"}
    page = _parameter_page
    await _mount_served_result(console_stub, page, descriptor, 390)
    await expect(page.get_by_label("Preview kind", exact=True)).to_have_text("text")
    await expect(page.get_by_role("combobox")).to_have_count(0)
    await expect(page.get_by_title("Preview kind", exact=True)).to_have_count(0)
    await expect(page.get_by_text("Fixed answer", exact=True)).to_be_visible()


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_router_restores_entry_view_and_forward_navigation(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """Back and forward restore the tab chosen for their own task."""
    page = _parameter_page
    await console_stub.seed_task("history-first", "google_flights", E2E_STARTED_AT, "completed", params={})
    await console_stub.seed_task("history-second", "google_flights", E2E_STARTED_AT, "completed", params={})
    await asyncio.to_thread(write_run_file, console_stub.data_dir, "history-first", "log", "First task log")
    await asyncio.to_thread(write_run_file, console_stub.data_dir, "history-second", "log", "Second task log")
    await page.goto(console_stub.base_url + "/task/history-first")
    await page.get_by_role("button", name="Log", exact=True).click()
    await expect(page.get_by_text("First task log", exact=False)).to_be_visible()
    await page.evaluate("import('/assets/js/router.js').then(r => r.navigate('/task/history-second'))")
    await page.get_by_role("button", name="Log", exact=True).click()
    await expect(page.get_by_text("Second task log", exact=False)).to_be_visible()
    await page.go_back()
    await expect(page.get_by_text("First task log", exact=False)).to_be_visible()
    await page.go_forward()
    await expect(page.get_by_text("Second task log", exact=False)).to_be_visible()


async def _expect_pdf_content(page: Page) -> None:
    """Wait for both colored marks and the text to appear."""
    await page.wait_for_function("""() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return false;
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      const minColor = 200, maxOther = 50, maxInk = 80;
      const minMarkShare = 0.05, minInkShare = 0.0001;
      let red = 0, blue = 0, ink = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (!pixels[i + 3]) continue;
        const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
        if (r > minColor && g < maxOther && b < maxOther) red++;
        if (b > minColor && r < maxOther && g < maxOther) blue++;
        if (r < maxInk && g < maxInk && b < maxInk) ink++;
      }
      const area = canvas.width * canvas.height;
      return red > area * minMarkShare && blue > area * minMarkShare && ink > area * minInkShare;
    }""")


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("width", [390, _WIDE_DESKTOP_VIEWPORT["width"]])
async def test_served_pdf_renders_and_remains_usable_after_zoom(
    console_stub: SimpleNamespace, _parameter_page: Page, pdf_bytes: bytes, width: int
) -> None:
    """A real PDF draws and remains readable after repeated zoom changes."""
    page = _parameter_page
    await asyncio.to_thread(write_share_file, console_stub.share, "upload/contract.pdf", pdf_bytes)
    await _served_surface(console_stub, page, width)
    await page.evaluate("""async () => {
      const { mountPdfViewer } = await import('/assets/js/pdf-viewer.js');
      const host = document.getElementById('surface');
      host.className = 'vp-pdf'; host.style.height = '600px';
      await mountPdfViewer(host, '/dav/upload/contract.pdf');
    }""")
    canvas = page.locator("canvas")
    await expect(page.get_by_text("Page 1 / 1", exact=True)).to_be_visible()
    await expect(canvas).to_be_visible()
    await _expect_pdf_content(page)
    initial = await canvas.bounding_box()
    await page.get_by_title("Zoom in", exact=True).click()
    await expect(page.get_by_text("125%", exact=True)).to_be_visible()
    await page.wait_for_function("""initial => {
      const canvas = document.querySelector('canvas');
      return canvas && canvas.getBoundingClientRect().width > initial;
    }""", arg=initial["width"])
    await _expect_pdf_content(page)
    await page.get_by_title("Zoom in", exact=True).click()
    await page.get_by_title("Zoom out", exact=True).click()
    await page.get_by_title("Fit to width", exact=True).click()
    await expect(page.get_by_text("100%", exact=True)).to_be_visible()
    await expect(canvas).to_be_visible()
    await expect(page.get_by_text("Couldn't render this page", exact=True)).to_have_count(0)
    await page.wait_for_function("""initial => {
      const canvas = document.querySelector('canvas');
      return canvas && Math.abs(canvas.getBoundingClientRect().width - initial) < 1;
    }""", arg=initial["width"])
    await _expect_pdf_content(page)


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_pdf_load_failure_offers_raw_file(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """A PDF load failure leaves an explanation and access to the raw file."""
    page = _parameter_page
    await asyncio.to_thread(write_share_file, console_stub.share, "upload/broken.pdf", b"invalid pdf")
    await _served_surface(console_stub, page, 390)
    await page.evaluate("""async () => {
      const { mountPdfViewer } = await import('/assets/js/pdf-viewer.js');
      await mountPdfViewer(document.getElementById('surface'), '/dav/upload/broken.pdf');
    }""")
    await expect(page.get_by_text("Couldn't load this PDF", exact=True)).to_be_visible()
    await expect(page.get_by_role("link", name="Open the raw file", exact=True)).to_have_attribute(
        "href", "/dav/upload/broken.pdf"
    )
    await expect(page.get_by_role("button")).to_have_count(0)


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("saved_tab,expected", [(None, "No result"), ("log", "Agent log content"), ("invalid", "No result")])
async def test_served_agent_tabs_restore_saved_choice(
    console_stub: SimpleNamespace, _parameter_page: Page, saved_tab: str | None, expected: str
) -> None:
    """Agent tabs open results by default and restore a valid saved choice."""
    page = _parameter_page
    await console_stub.seed_task("agent-contract", "google_flights", E2E_STARTED_AT, "completed", params={})
    await asyncio.to_thread(write_run_file, console_stub.data_dir, "agent-contract", "log", "Agent log content")
    await _served_surface(console_stub, page, 390)
    await page.evaluate("""async (tab) => {
      const { mountAgent } = await import('/assets/js/screens/agent-detail.js');
      const detail = await fetch('/api/tasks/agent-contract').then(res => res.json());
      detail.id = 'agent-contract';
      window.agentHandle = mountAgent(document.getElementById('surface'), detail, { tab });
    }""", saved_tab)
    tabs = page.get_by_role("button", name=re.compile(r"^(Results|Conversation|Log|Tool calls)$"))
    assert await tabs.all_text_contents() == ["Results", "Conversation", "Log", "Tool calls"]
    await expect(page.get_by_text(expected, exact=False)).to_be_visible()
    await page.get_by_role("button", name="Tool calls", exact=True).click()
    await expect(page.get_by_text("No subtasks", exact=False)).to_be_visible()
    await page.get_by_role("button", name="Log", exact=True).click()
    await expect(page.get_by_text("Agent log content", exact=False)).to_be_visible()


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("width", [390, _WIDE_DESKTOP_VIEWPORT["width"]])
async def test_served_image_viewer_fits_and_zooms(
    console_stub: SimpleNamespace, _parameter_page: Page, width: int
) -> None:
    """An image fits the available space and responds to zoom controls."""
    page = _parameter_page
    await asyncio.to_thread(write_share_file, console_stub.share, "upload/contract.png", E2E_PNG)
    await _served_surface(console_stub, page, width)
    await page.evaluate("""async () => {
      const { renderView } = await import('/assets/js/viewer.js');
      renderView({ path: 'upload/contract.png', name: 'contract.png', mode: 'native',
        kind: 'image', url: '/dav/upload/contract.png' }, document.getElementById('surface'));
    }""")
    image = page.locator("img")
    await page.wait_for_function("() => document.querySelector('img').naturalWidth > 0")
    initial = await image.bounding_box()
    assert initial["x"] >= 0
    assert initial["x"] + initial["width"] <= width
    await page.get_by_title("Zoom in", exact=True).click()
    zoomed = await image.bounding_box()
    assert zoomed["width"] > initial["width"]
    await page.get_by_title("Fit to stage", exact=True).click()
    fitted = await image.bounding_box()
    assert fitted["width"] == initial["width"]
    assert fitted["height"] == initial["height"]


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_served_image_load_failure_keeps_download(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """An image that cannot display keeps a usable download action."""
    page = _parameter_page
    await asyncio.to_thread(write_share_file, console_stub.share, "upload/broken.png", b"invalid image")
    await _served_surface(console_stub, page, 390)
    await page.evaluate("""async () => {
      const { renderView } = await import('/assets/js/viewer.js');
      renderView({ path: 'upload/broken.png', name: 'broken.png', mode: 'native',
        kind: 'image', url: '/dav/upload/broken.png' }, document.getElementById('surface'));
    }""")
    await expect(page.get_by_text("Can't display this image", exact=True)).to_be_visible()
    async with page.expect_download() as downloaded:
        await page.get_by_role("link", name="Download", exact=True).click()
    download = await downloaded.value
    assert download.suggested_filename == "broken.png"
    path = await download.path()
    assert await asyncio.to_thread(Path(path).read_bytes) == b"invalid image"


_FRONTEND_FILE_SIZE = 1024


_PARAMETER_PHONE_HEIGHT = _MOBILE_VIEWPORT["height"]


@pytest.fixture
async def _parameter_page(tmp_path: Path) -> AsyncIterator[Page]:
    """Open an isolated page for parameter interactions."""
    executable = Path(os.environ.get("PARAMETER_UI_BROWSER", binary_info()["binary_path"]))
    assert executable.is_file(), f"Browser executable is missing at {executable}"
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(executable_path=str(executable), headless=True, args=["--no-sandbox"])
        try:
            context = await browser.new_context(viewport={"width": 1280, "height": _WIDE_DESKTOP_VIEWPORT["height"]})
            try:
                page = await context.new_page()
                try:
                    yield page
                finally:
                    if not page.is_closed():
                        await page.screenshot(path=str(tmp_path / "parameter-ui.png"), full_page=True)
            finally:
                await context.close()
        finally:
            await browser.close()


def _parameter_schema(params: list[dict]) -> dict:
    return {"name": "google_flights", "summary": "Parameter demo", "description": "", "rules": [], "params": [
        {"required": True, "default": None, "description": "", "enum": None, "min": None, "max": None,
         "step": None, "item_type": None, "kind": None, **param} for param in params
    ]}


async def _parameter_form(stub: SimpleNamespace, page: Page, values: list[str]) -> None:
    stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "items", "type": "array", "item_type": "string"},
        {"name": "context", "type": "string", "kind": "text"},
    ]))
    await stub.seed_task("parameter-source", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                         params={"items": values, "context": "unchanged"})
    await page.goto(stub.base_url + "/tools/google_flights/run?from=parameter-source")
    await page.get_by_role("textbox", name="items item", exact=True).wait_for()
    await page.wait_for_function("() => document.fonts.status === 'loaded'")


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_waits_for_saved_values(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Run waits for the form and sends the saved values."""
    page = _parameter_page
    schema_pending, schema_release = asyncio.Event(), asyncio.Event()
    source_pending, source_release = asyncio.Event(), asyncio.Event()
    saved = {"items": ["first", "second"], "query": "saved query"}
    schema = _parameter_schema([
        {"name": "items", "type": "array", "item_type": "string"},
        {"name": "query", "type": "string", "kind": "text"},
    ])
    schema["description"] = "Saved run details"
    calls = []

    async def hold_schema(route: PlaywrightRoute) -> None:
        schema_pending.set()
        await schema_release.wait()
        await route.fulfill(json=schema)

    async def hold_source(route: PlaywrightRoute) -> None:
        source_pending.set()
        await source_release.wait()
        await route.fulfill(json={"task": {"params": saved}})

    async def capture_call(route: PlaywrightRoute) -> None:
        calls.append(route.request.post_data_json)
        await route.fulfill(json={"result": "Saved values received"})

    await page.route("**/api/tools/google_flights/schema", hold_schema)
    await page.route("**/api/tasks/parameter-pending", hold_source)
    await page.route("**/api/tools/google_flights/call", capture_call)
    try:
        await page.goto(console_stub.base_url + "/tools/google_flights/run?from=parameter-pending")
        await asyncio.wait_for(schema_pending.wait(), timeout=10)
        run = page.get_by_role("button", name="Run tool", exact=True)
        await expect(run).to_be_disabled()
        schema_release.set()
        await asyncio.wait_for(source_pending.wait(), timeout=10)
        await expect(run).to_be_disabled()
        await run.click(force=True)
        await page.get_by_text("Saved run details", exact=True).dispatch_event("keydown", {"key": "Enter", "bubbles": True})
        assert calls == []
        source_release.set()
        await expect(run).to_be_enabled()
        await expect(page.get_by_role("textbox", name="query", exact=True)).to_have_value("saved query")
        await expect(page.get_by_role("button", name="Edit first", exact=True)).to_be_visible()
        await run.click()
        await expect(page.get_by_text("Saved values received", exact=True)).to_be_visible()
        assert calls == [saved]
    finally:
        schema_release.set()
        source_release.set()
        await page.unroute_all(behavior="wait")


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("width", [1280, 390])
async def test_parameter_ui_edits_preserve_submitted_values(console_stub: SimpleNamespace, _parameter_page: Page, width: int) -> None:
    """Edits and drafts preserve hidden values and exact submitted order."""
    page = _parameter_page
    await page.set_viewport_size({"width": width, "height": _WIDE_DESKTOP_VIEWPORT["height"]})
    await _parameter_form(console_stub, page, ["A", "B", "C", "D", "E", "F", "HIDDEN"])
    await page.get_by_role("button", name="Edit B", exact=True).click()
    await page.get_by_role("textbox", name="Edit items item", exact=True).fill("B\nsaved")
    await page.get_by_role("button", name="Save", exact=True).click()
    await page.get_by_role("button", name="Edit A", exact=True).click()
    await page.get_by_role("textbox", name="Edit items item", exact=True).fill("discarded")
    await page.get_by_role("group", name="items", exact=True).get_by_role("button", name="Cancel", exact=True).click()
    await page.get_by_role("button", name="Edit C", exact=True).click()
    await page.get_by_role("button", name="Remove C", exact=True).click()
    composer = page.get_by_role("textbox", name="items item", exact=True)
    await composer.fill("ADD")
    await page.get_by_role("button", name="Add", exact=True).click()
    await composer.fill("RUN\ndraft")
    async with page.expect_request(lambda request: request.url.endswith("/google_flights/call")) as request:
        await page.get_by_role("button", name="Run tool", exact=True).click()
    expected = {"items": ["A", "B\nsaved", "D", "E", "F", "HIDDEN", "ADD", "RUN\ndraft"], "context": "unchanged"}
    assert (await request.value).post_data_json == expected
    await page.wait_for_url("**/task/*")
    async with httpx.AsyncClient(base_url=console_stub.base_url) as client:
        response = await client.get("/api/tasks/" + console_stub.last_call_id())
    assert response.json()["task"]["params"] == expected


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("value", ["first\r\nsecond", "first\rsecond", " \tfirst\nsecond\t "],
                         ids=["crlf", "cr", "outer-whitespace"])
async def test_parameter_ui_unchanged_list_edit_preserves_value(
    console_stub: SimpleNamespace, _parameter_page: Page, value: str,
) -> None:
    """Saving an unchanged item preserves its exact submitted value."""
    page = _parameter_page
    await _parameter_form(console_stub, page, [value])
    await page.get_by_role("group", name="items", exact=True).get_by_role("button", name=re.compile("^Edit ")).click()
    await page.get_by_role("textbox", name="Edit items item", exact=True).wait_for()
    await page.get_by_role("button", name="Save", exact=True).click()
    await _parameter_submit(page, {"items": [value], "context": "unchanged"})


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("action", ["Save", "Cancel"])
async def test_parameter_ui_request_failure_keeps_edit_guard(
    console_stub: SimpleNamespace, _parameter_page: Page, action: str,
) -> None:
    """A failed request keeps Run blocked until the open edit ends."""
    page = _parameter_page
    await _parameter_form(console_stub, page, ["A", "B"])
    pending = asyncio.Event()
    release = asyncio.Event()

    async def fail_call(route: PlaywrightRoute) -> None:
        pending.set()
        await release.wait()
        await route.fulfill(status=500, json={"code": "stub_down", "error": "Request recovery test"})

    await page.route("**/api/tools/google_flights/call", fail_call)
    try:
        await page.get_by_role("button", name="Run tool", exact=True).click()
        await asyncio.wait_for(pending.wait(), timeout=10)
        await page.get_by_role("button", name="Edit A", exact=True).click()
        editor = page.get_by_role("textbox", name="Edit items item", exact=True)
        await editor.fill("A changed")
        release.set()
        await expect(page.get_by_role("alert").filter(has_text="Request recovery test")).to_be_visible()
        run = page.get_by_role("button", name="Run tool", exact=True)
        await expect(run).to_be_visible()
        await expect(editor).to_have_value("A changed")
        await expect(run).to_be_disabled()
        await page.get_by_role("group", name="items", exact=True).get_by_role("button", name=action, exact=True).click()
        await expect(editor).to_have_count(0)
        await expect(run).to_be_enabled()
    finally:
        release.set()
        await page.unroute_all(behavior="wait")
    expected = ["A changed", "B"] if action == "Save" else ["A", "B"]
    await _parameter_submit(page, {"items": expected, "context": "unchanged"})


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("action", ["Cancel", "Remove Short item"], ids=["cancel", "remove"])
@pytest.mark.parametrize("composer_error", [False, True], ids=["editor-error", "composer-and-editor-errors"])
async def test_parameter_ui_rejected_edit_recovery(
    console_stub: SimpleNamespace, _parameter_page: Page, action: str, composer_error: bool,
) -> None:
    """Discarding an invalid edit clears its error and keeps the remaining values."""
    page = _parameter_page
    await _parameter_form(console_stub, page, ["Short item", "A second item"])
    group = page.get_by_role("group", name="items", exact=True)
    composer = page.get_by_role("textbox", name="items item", exact=True)
    error = group.get_by_text("This item is already in the list.", exact=True)
    if composer_error:
        await composer.fill("A second item")
        await group.get_by_role("button", name="Add", exact=True).click()
        await expect(error).to_have_count(1)
    await page.get_by_role("button", name="Edit Short item", exact=True).click()
    editor = page.get_by_role("textbox", name="Edit items item", exact=True)
    await editor.fill("A second item")
    await group.get_by_role("button", name="Save", exact=True).click()
    await expect(error).to_have_count(2 if composer_error else 1)
    await expect(page.get_by_role("button", name="Run tool", exact=True)).to_be_disabled()
    assert console_stub.last_call_id() is None
    await group.get_by_role("button", name=action, exact=True).click()
    await expect(editor).to_have_count(0)
    await expect(error).to_have_count(1 if composer_error else 0)
    await expect(page.get_by_role("button", name="Run tool", exact=True)).to_be_enabled()
    if action == "Cancel":
        await expect(page.get_by_role("button", name="Edit Short item", exact=True)).to_be_focused()
        expected = ["Short item", "A second item"]
    else:
        await expect(page.get_by_role("button", name="Edit Short item", exact=True)).to_have_count(0)
        await expect(page.get_by_role("button", name="Edit A second item", exact=True)).to_be_focused()
        expected = ["A second item"]
    if composer_error:
        await composer.fill("")
        await expect(error).to_have_count(0)
    await _parameter_submit(page, {"items": expected, "context": "unchanged"})


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("editor_error", [False, True], ids=["composer-error", "composer-and-editor-errors"])
async def test_parameter_ui_composer_escape_clears_discarded_error(
    console_stub: SimpleNamespace, _parameter_page: Page, editor_error: bool,
) -> None:
    """Escape discards the composer error and preserves saved items and other errors."""
    page = _parameter_page
    await _parameter_form(console_stub, page, ["A", "B"])
    group = page.get_by_role("group", name="items", exact=True)
    composer = page.get_by_role("textbox", name="items item", exact=True)
    error = group.get_by_text("This item is already in the list.", exact=True)
    await composer.fill("A")
    await group.get_by_role("button", name="Add", exact=True).click()
    await expect(error).to_have_count(1)
    if editor_error:
        await group.get_by_role("button", name="Edit B", exact=True).click()
        await group.get_by_role("textbox", name="Edit items item", exact=True).fill("A")
        await group.get_by_role("button", name="Save", exact=True).click()
        await expect(error).to_have_count(2)
    await composer.press("Escape")
    await expect(composer).to_have_value("")
    await expect(composer).to_be_focused()
    await expect(group.get_by_role("button", name="Add", exact=True)).to_be_disabled()
    await expect(error).to_have_count(1 if editor_error else 0)
    await composer.press("Escape")
    await expect(composer).not_to_be_focused()
    if editor_error:
        await expect(page.get_by_role("button", name="Run tool", exact=True)).to_be_disabled()
        await group.get_by_role("button", name="Cancel", exact=True).click()
    await expect(error).to_have_count(0)
    await expect(group.get_by_role("button", name="Edit A", exact=True)).to_be_visible()
    await expect(group.get_by_role("button", name="Edit B", exact=True)).to_be_visible()
    await _parameter_submit(page, {"items": ["A", "B"], "context": "unchanged"})


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_list_accessibility_context(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """List composers keep required status and help through errors and recovery."""
    page = _parameter_page
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "items", "type": "array", "item_type": "string", "description": "Choose the destinations."},
        {"name": "extras", "type": "array", "item_type": "string", "required": False, "description": "Extra destinations."},
    ]))
    await page.goto(console_stub.base_url + "/tools/google_flights/run")
    composer = page.get_by_role("textbox", name="items item", exact=True)
    optional = page.get_by_role("textbox", name="extras item", exact=True)
    await expect(composer).to_have_attribute("aria-required", "true")
    await expect(composer).to_have_accessible_description("Choose the destinations.")
    await expect(optional).to_have_attribute("aria-required", "false")
    await expect(optional).to_have_accessible_description("Extra destinations.")
    await page.get_by_role("button", name="Run tool", exact=True).click()
    await expect(composer).to_be_focused()
    await expect(composer).to_have_accessible_description("Choose the destinations. items is required.")
    await expect(composer).to_have_attribute("aria-required", "true")
    assert console_stub.last_call_id() is None
    await composer.fill("Paris")
    await page.get_by_role("group", name="items", exact=True).get_by_role("button", name="Add", exact=True).click()
    await expect(composer).to_have_accessible_description("Choose the destinations.")
    await expect(composer).to_have_attribute("aria-required", "true")
    await composer.press("Backspace")
    await page.get_by_role("button", name="Run tool", exact=True).click()
    await expect(composer).to_have_accessible_description("Choose the destinations. items is required.")
    await composer.fill("London")
    await page.get_by_role("group", name="items", exact=True).get_by_role("button", name="Add", exact=True).click()
    await expect(composer).to_have_accessible_description("Choose the destinations.")
    await expect(optional).to_have_accessible_description("Extra destinations.")
    await _parameter_submit(page, {"items": ["London"]})


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_keyboard_and_edit_guard(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Keyboard drafts and direct edit decisions keep focus and prevent early runs."""
    page = _parameter_page
    await _parameter_form(console_stub, page, ["A", "B"])
    edit = page.get_by_role("button", name="Edit A", exact=True)
    await edit.focus()
    await page.keyboard.press("Enter")
    editor = page.get_by_role("textbox", name="Edit items item", exact=True)
    assert await editor.input_value() == "A"
    assert await editor.evaluate("node => node === document.activeElement")
    assert await page.get_by_role("button", name="Run tool", exact=True).is_disabled()
    assert console_stub.last_call_id() is None
    await page.get_by_role("group", name="items", exact=True).get_by_role("button", name="Cancel", exact=True).click()
    assert await edit.evaluate("node => node === document.activeElement")
    composer = page.get_by_role("textbox", name="items item", exact=True)
    await composer.fill("line")
    await composer.press("End")
    await composer.press("Enter")
    await composer.press("x")
    assert await composer.input_value() == "line\nx"
    await composer.press("Escape")
    assert await composer.input_value() == ""
    await composer.press("Backspace")
    assert await page.get_by_role("button", name="Edit B", exact=True).count() == 0
    assert await composer.evaluate("node => node === document.activeElement")
    await composer.press("Escape")
    assert not await composer.evaluate("node => node === document.activeElement")


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_overflow_is_readable(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Overflowing saved text expands to its complete readable height."""
    page = _parameter_page
    text = "A complete long line with spaces. " * 90
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "items", "type": "array", "item_type": "string"},
        {"name": "context", "type": "string", "kind": "longtext"},
        {"name": "default_text", "type": "string", "kind": "text", "default": text, "required": False},
    ]))
    await _parameter_form(console_stub, page, ["short", text])
    group = page.get_by_role("group", name="items", exact=True)
    more = group.get_by_role("button", name="Read more", exact=True)
    await more.wait_for()
    assert await more.count() == 1
    value = group.get_by_text(text, exact=True)
    collapsed = await value.bounding_box()
    await more.click()
    assert await group.get_by_role("button", name="Read less").get_attribute("aria-expanded") == "true"
    assert await value.inner_text() == text
    assert await value.evaluate("node => node.scrollHeight <= node.clientHeight + 1")
    assert (await value.bounding_box())["height"] > collapsed["height"]
    await page.set_viewport_size(dict(_MOBILE_VIEWPORT))
    await group.get_by_role("button", name="Read less").click()
    await more.wait_for()
    assert await value.evaluate("node => node.scrollHeight > node.clientHeight")
    await console_stub.seed_task("parameter-overflow", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                params={"context": text, "raw": {"stored": text}, "default_text": text})
    await page.goto(console_stub.base_url + "/task/parameter-overflow")
    scalar = page.get_by_role("group", name="context parameter", exact=True)
    await scalar.get_by_role("button", name="Read more", exact=True).click()
    assert await scalar.get_by_text(text, exact=True).evaluate("node => node.scrollHeight <= node.clientHeight + 1")
    raw = page.get_by_role("group", name="raw parameter", exact=True)
    await raw.get_by_role("button", name="Read more", exact=True).click()
    assert json.loads(await raw.get_by_text(json.dumps({"stored": text}, indent=2), exact=True).inner_text()) == {"stored": text}
    await page.get_by_role("button", name="1 param at default · show", exact=True).click()
    default = page.get_by_role("group", name="default_text parameter", exact=True)
    await default.get_by_role("button", name="Read more", exact=True).wait_for()


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_bounded_list_retains_identity(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """A bounded collection retains duplicate positions and newly added tail values."""
    page = _parameter_page
    values = ["duplicate", "duplicate"] + [f"Value {index}" for index in range(9999)]
    await _parameter_form(console_stub, page, values)
    group = page.get_by_role("group", name="items", exact=True)
    assert await group.get_by_role("button", name=re.compile("^Edit ")).count() == 6
    await group.get_by_role("button", name="Edit duplicate", exact=True).nth(1).click()
    await group.get_by_role("textbox", name="Edit items item").fill("changed duplicate")
    await group.get_by_role("button", name="Save", exact=True).click()
    await group.get_by_role("button", name="Show next 50").click()
    assert await group.get_by_role("button", name=re.compile("^Edit ")).count() == 56
    await group.get_by_role("button", name="Edit Value 8", exact=True).click()
    assert await group.get_by_role("button", name="Show first 6").is_disabled()
    await group.get_by_role("button", name="Cancel", exact=True).click()
    await group.get_by_role("button", name="Show first 6").click()
    await group.get_by_role("textbox", name="items item", exact=True).fill("tail addition")
    await group.get_by_role("button", name="Add", exact=True).click()
    assert await group.get_by_role("button", name="Edit tail addition", exact=True).count() == 1
    async with page.expect_request(lambda request: request.url.endswith("/google_flights/call")) as request:
        await page.get_by_role("button", name="Run tool", exact=True).click()
    values[1] = "changed duplicate"
    assert (await request.value).post_data_json["items"] == values + ["tail addition"]


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_phone_action_geometry(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Phone actions keep a fixed reachable anchor without overlapping text."""
    page = _parameter_page
    await page.set_viewport_size(dict(_MOBILE_VIEWPORT))
    await _parameter_form(console_stub, page, ["short", "Long readable value. " * 60])
    edit = page.get_by_role("button", name="Edit short", exact=True)
    before = await edit.bounding_box()
    assert before["width"] >= 44
    assert before["height"] >= 44
    await edit.click()
    trash = await page.get_by_role("button", name="Remove short", exact=True).bounding_box()
    assert abs(before["x"] - trash["x"]) <= 1
    editor = await page.get_by_role("textbox", name="Edit items item").bounding_box()
    assert editor["x"] + editor["width"] <= trash["x"]
    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_values_have_no_copy_actions(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Parameter interaction does not copy values and task metadata remains copyable."""
    page = _parameter_page
    await page.add_init_script("window.copied=[]; navigator.clipboard.writeText=async text=>window.copied.push(text); document.execCommand=()=>{window.copied.push('legacy');return true;}")
    await _parameter_form(console_stub, page, ["ordinary/path.txt"])
    assert await page.get_by_role("group", name="items", exact=True).get_by_role("button", name=re.compile("Copy")).count() == 0
    await page.goto(console_stub.base_url + "/task/parameter-source")
    await expect(page.locator(".td-params").get_by_role("button", name=re.compile("Copy"))).to_have_count(0)
    value = page.get_by_text("ordinary/path.txt", exact=True)
    await value.click()
    assert await page.evaluate("window.copied") == []
    assert await value.get_attribute("href") is None
    copy = page.get_by_role("button", name=re.compile("Copy.*ID|Copy.*id"))
    await copy.click()
    assert await page.evaluate("window.copied") == ["parameter-source"]


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_schema_controls_and_raw_fallback(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Declared controls retain constraints and unsupported saved data stays faithful."""
    page = _parameter_page
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "choice", "type": "string", "kind": "text", "enum": ["one", "two"]},
        {"name": "amount", "type": "integer", "min": 2, "max": 4},
        {"name": "day_from", "type": "string", "kind": "date"},
        {"name": "day_to", "type": "string", "kind": "date"},
        {"name": "items", "type": "array", "item_type": "string"},
    ]))
    await console_stub.seed_task("parameter-raw", "google_flights", "2026-10-31T01:00:00+00:00", "completed", params={"items": ["keep", {"exact": 1}, None]})
    await page.goto(console_stub.base_url + "/tools/google_flights/run?from=parameter-raw")
    await page.get_by_role("combobox", name="choice", exact=True).select_option("two")
    await page.get_by_role("spinbutton", name="amount", exact=True).fill("1")
    await page.get_by_label("day_from", exact=True).fill("2026-10-31")
    await page.get_by_label("day_to", exact=True).fill("2026-11-01")
    await page.get_by_role("button", name="Run tool", exact=True).click()
    await page.get_by_text("amount must be ≥ 2.", exact=True).wait_for()
    await page.get_by_text("items has an unsupported saved value.", exact=True).wait_for()
    assert console_stub.last_call_id() is None
    assert json.loads(await page.get_by_text(json.dumps(["keep", {"exact": 1}, None], indent=2), exact=True).inner_text()) == ["keep", {"exact": 1}, None]


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
@pytest.mark.parametrize("declaration,value", [
    ({"kind": "text", "enum": ["one", "two"]}, "obsolete choice"),
    ({"kind": "date"}, "not a date"),
    ({"kind": "date"}, "2026-02-30"),
    ({"kind": "text"}, "first\nsecond"),
    ({"kind": "text"}, "first\r\nsecond"),
    ({"kind": "text"}, " padded "),
    ({"kind": "longtext"}, "first\r\nsecond"),
    ({"kind": "longtext"}, " padded "),
], ids=["obsolete-enum", "invalid-date", "impossible-date", "multiline-text", "crlf-text",
        "padded-text", "crlf-longtext", "padded-longtext"])
async def test_parameter_ui_unrepresentable_saved_values(
    console_stub: SimpleNamespace, _parameter_page: Page, required: bool, declaration: dict, value: str,
) -> None:
    """Saved values that a control cannot show stay visible and block submission."""
    page = _parameter_page
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "saved", "type": "string", "required": required, **declaration},
    ]))
    await console_stub.seed_task("parameter-unrepresentable", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                params={"saved": value})
    await page.goto(console_stub.base_url + "/tools/google_flights/run?from=parameter-unrepresentable")
    field = page.locator('[data-parameter-field="saved"]')
    await field.get_by_text(value, exact=True).wait_for()
    await page.get_by_role("button", name="Run tool", exact=True).click()
    await field.get_by_text("saved has an unsupported saved value.", exact=True).wait_for()
    assert await field.get_by_text(value, exact=True).inner_text() == value
    assert console_stub.last_call_id() is None
    assert page.url == console_stub.base_url + "/tools/google_flights/run?from=parameter-unrepresentable"


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
async def test_parameter_ui_phone_incompatible_boolean_prefill(
    console_stub: SimpleNamespace, _parameter_page: Page, required: bool,
) -> None:
    """An unsupported saved boolean stays visible without a switch or false value."""
    page = _parameter_page
    await page.set_viewport_size({"width": 390, "height": _PARAMETER_PHONE_HEIGHT})
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "flag", "type": "boolean", "required": required, "description": "Keep the saved flag."},
    ]))
    await console_stub.seed_task("parameter-boolean", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                params={"flag": "invalid saved bool"})
    url = console_stub.base_url + "/tools/google_flights/run?from=parameter-boolean"
    await page.goto(url)
    if not required:
        await page.get_by_role("button", name="Expand", exact=True).click()
    await page.get_by_text("invalid saved bool", exact=True).wait_for()
    assert await page.get_by_text("Keep the saved flag.", exact=True).is_visible()
    assert not await page.get_by_role("switch", name="flag", exact=True).is_visible()
    assert not await page.get_by_text("false", exact=True).is_visible()
    assert not await page.get_by_text("true", exact=True).is_visible()
    await page.get_by_role("button", name="Run tool", exact=True).click()
    await page.get_by_text("flag has an unsupported saved value.", exact=True).wait_for()
    assert await page.get_by_text("invalid saved bool", exact=True).is_visible()
    assert console_stub.last_call_id() is None
    assert page.url == url


async def _parameter_submit(page: Page, expected: dict) -> None:
    """Run the tool and check its submitted values."""
    async with page.expect_request(lambda request: request.url.endswith("/google_flights/call")) as request:
        await page.get_by_role("button", name="Run tool", exact=True).click()
    assert (await request.value).post_data_json == expected
    await page.wait_for_url("**/task/*")


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("kind,default,saved,edited", [
    ("text", " padded ", "valid text", "edited text"),
    ("longtext", "first\r\nsecond", "valid\ntext", "edited\ntext"),
    ("date", "2026-02-30", "2026-11-01", "2026-11-02"),
    ("enum", "obsolete choice", "one", "two"),
])
async def test_parameter_ui_saved_value_overrides_incompatible_default(
    console_stub: SimpleNamespace, _parameter_page: Page, kind: str, default: str, saved: str, edited: str,
) -> None:
    """Saved values replace unsupported defaults and keep their controls editable."""
    page = _parameter_page
    declaration = {"name": "query", "type": "string", "kind": "text" if kind == "enum" else kind,
                   "default": default}
    if kind == "enum":
        declaration["enum"] = ["one", "two"]
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([declaration]))
    await console_stub.seed_task("parameter-override", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                params={"query": saved})
    url = console_stub.base_url + "/tools/google_flights/run?from=parameter-override"
    await page.goto(url)
    control = page.get_by_label("query", exact=True)
    await control.wait_for()
    assert await control.input_value() == saved
    assert not await page.get_by_text("query has an unsupported saved value.", exact=True).is_visible()
    await _parameter_submit(page, {"query": saved})
    await page.goto(url)
    if kind == "enum":
        await control.select_option(edited)
    else:
        await control.fill(edited)
    assert await control.input_value() == edited
    await _parameter_submit(page, {"query": edited})


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("kind", [None, "resource_list", "resource"], ids=["ordinary", "resource-list", "resource"])
@pytest.mark.parametrize("source", ["default", "prefill"])
async def test_parameter_ui_empty_resources_and_lists_stay_editable(
    console_stub: SimpleNamespace, _parameter_page: Page, kind: str | None, source: str,
) -> None:
    """Empty resources and lists allow submission and accept new values afterward."""
    page = _parameter_page
    empty = "" if kind == "resource" else []
    declaration = {"name": "items", "type": "string" if kind == "resource" else "array",
                   "item_type": None if kind == "resource" else "string", "kind": kind, "required": False}
    if source == "default":
        declaration["default"] = empty
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([declaration]))
    url = console_stub.base_url + "/tools/google_flights/run"
    if source == "prefill":
        await console_stub.seed_task("parameter-empty", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                    params={"items": empty})
        url += "?from=parameter-empty"
    await page.goto(url)
    field = page.locator('[data-parameter-field="items"]')
    choose = "Choose file…" if kind == "resource" else "Add resource…"
    if kind in {"resource", "resource_list"}:
        await field.get_by_role("button", name=choose, exact=True).wait_for()
    else:
        await field.get_by_role("textbox", name="items item", exact=True).wait_for()
    await _parameter_submit(page, {})
    await page.goto(url)
    field = page.locator('[data-parameter-field="items"]')
    if kind in {"resource", "resource_list"}:
        write_share_file(console_stub.share, "upload/sample.png", E2E_PNG)
        fact = {"name": "sample.png", "path": "upload/sample.png", "type": "file", "size": len(E2E_PNG),
                "kind": "image", "mtime": 0, "thumb": {"status": "done", "url": "/dav/upload/sample.png"}}
        console_stub.pin_body(r"/api/resources$", {"path": "upload", "entries": [fact]})
        hint = "No file selected." if kind == "resource" else "No files selected."
        await field.get_by_text(hint, exact=True).wait_for()
        await field.get_by_role("button", name=choose, exact=True).click()
        await field.get_by_text("sample.png", exact=True).click()
        await field.get_by_role("button", name="Choose" if kind == "resource" else "Add (1)", exact=True).filter(visible=True).click()
        await field.get_by_role("link", name="Open sample.png in a new tab", exact=True).wait_for()
        value = "upload/sample.png"
    else:
        value = "new item"
        await field.get_by_role("textbox", name="items item", exact=True).fill(value)
        await field.get_by_role("button", name="Add", exact=True).click()
        await field.get_by_role("button", name="Edit new item", exact=True).wait_for()
    await _parameter_submit(page, {"items": value if kind == "resource" else [value]})


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("kind,value", [("text", "new text"), ("longtext", "first\nsecond"),
                                       ("date", "2026-11-01"), ("enum", "two")])
@pytest.mark.parametrize("source", ["default", "prefill"])
async def test_parameter_ui_empty_strings_stay_editable(
    console_stub: SimpleNamespace, _parameter_page: Page, kind: str, value: str, source: str,
) -> None:
    """Empty strings allow submission and keep their native controls editable."""
    page = _parameter_page
    declaration = {"name": "query", "type": "string", "kind": "text" if kind == "enum" else kind, "required": False}
    if kind == "enum":
        declaration["enum"] = ["one", "two"]
    if source == "default":
        declaration["default"] = ""
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([declaration]))
    url = console_stub.base_url + "/tools/google_flights/run"
    if source == "prefill":
        await console_stub.seed_task("parameter-empty-string", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                    params={"query": ""})
        url += "?from=parameter-empty-string"
    await page.goto(url)
    control = page.get_by_label("query", exact=True)
    await control.wait_for()
    assert await control.input_value() == ""
    await _parameter_submit(page, {})
    await page.goto(url)
    control = page.get_by_label("query", exact=True)
    if kind == "enum":
        await control.select_option(value)
    else:
        await control.fill(value)
    assert await control.input_value() == value
    await _parameter_submit(page, {"query": value})


async def _parameter_resource_alignment(page: Page, name: str) -> None:
    """Check the resource starts below its label."""
    label = await page.locator(f'[data-parameter-label="{name}"]').bounding_box()
    value = await page.locator(f'[data-parameter-value="{name}"]').bounding_box()
    assert abs(value["x"] - label["x"]) <= 1
    assert value["y"] >= label["y"] + label["height"]
    viewport_width = await page.evaluate("document.documentElement.clientWidth")
    assert value["x"] >= 0
    assert value["x"] + value["width"] <= viewport_width


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_native_resource_workflow(console_stub: SimpleNamespace, _parameter_page: Page) -> None:
    """Native resources retain previews, viewer links, removal and collection behavior."""
    page = _parameter_page
    write_share_file(console_stub.share, "upload/sample.png", E2E_PNG)
    params = [
        {"name": "image", "type": "string", "kind": "resource"},
        {"name": "images", "type": "array", "item_type": "string", "kind": "resource_list"},
        {"name": "unset", "type": "string", "kind": "resource", "required": False},
    ]
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema(params))
    fact = {"name": "sample.png", "path": "upload/sample.png", "type": "file", "size": len(E2E_PNG),
            "kind": "image", "mtime": 0, "thumb": {"status": "done", "url": "/dav/upload/sample.png"}}
    console_stub.pin_body(r"/api/resources$", {"path": "upload", "entries": [fact]})
    await console_stub.seed_task("parameter-files", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                params={"image": "upload/sample.png", "images": ["upload/sample.png"] * 7, "unset": None})
    await page.goto(console_stub.base_url + "/task/parameter-files")
    link = page.get_by_role("link", name="Open sample.png in a new tab", exact=True).first
    await link.wait_for()
    await page.get_by_role("button", name="1 param at default · show", exact=True).click()
    await _parameter_resource_alignment(page, "image")
    await _parameter_resource_alignment(page, "images")
    await _parameter_resource_alignment(page, "unset")
    await page.wait_for_function("Array.from(document.images).some(image => image.naturalWidth > 0)")
    assert await link.get_by_text("sample.png", exact=True).get_attribute("title") == "upload/sample.png"
    assert "image" in await link.inner_text()
    assert await page.get_by_role("button", name=re.compile("Copy (value|path)")).count() == 0
    async with page.expect_popup() as opened:
        await link.click()
    popup = await opened.value
    await popup.wait_for_url("**/view/upload/sample.png")
    await popup.wait_for_function("Array.from(document.images).some(image => image.naturalWidth > 0)")
    await popup.get_by_role("button", name="Fit", exact=True).wait_for()
    await popup.close()
    summary = page.get_by_role("button", name="Expand value of images", exact=True)
    await summary.click()
    assert await page.get_by_role("link", name="Open sample.png in a new tab", exact=True).count() == 8
    await page.goto(console_stub.base_url + "/tools/google_flights/run?from=parameter-files")
    field = page.locator('[data-parameter-field="image"]')
    await field.get_by_role("button", name="Choose file…", exact=True).wait_for()
    await field.get_by_role("button", name="Remove", exact=True).click()
    await field.get_by_text("No file selected.", exact=True).wait_for()
    await field.get_by_role("button", name="Choose file…", exact=True).click()
    await field.get_by_text("Choose from Resources", exact=True).filter(visible=True).wait_for()
    await field.get_by_text("sample.png", exact=True).click()
    await field.get_by_role("button", name="Choose", exact=True).click()
    await page.get_by_role("link", name="Open sample.png in a new tab", exact=True).first.wait_for()
    assert await page.get_by_role("button", name=re.compile("Edit sample|Read more")).count() == 0
    await console_stub.seed_task("parameter-gone", "google_flights", "2026-10-31T01:00:00+00:00", "completed",
                                params={"image": "upload/missing.png", "images": [], "unset": None})
    await page.goto(console_stub.base_url + "/task/parameter-gone")
    await page.get_by_text("Removed by 30-day retention", exact=False).wait_for()
    assert await page.get_by_role("link", name="Open missing.png in a new tab", exact=True).count() == 0


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("schema_available", [False, True])
async def test_parameter_ui_missing_metadata_uses_value_shape(
    console_stub: SimpleNamespace, _parameter_page: Page, schema_available: bool,
) -> None:
    """Saved values stay readable when their declaration is missing."""
    page = _parameter_page
    if schema_available:
        console_stub.pin_body(r"/api/tools/collect_evidence/schema$", _parameter_schema([
            {"name": "declared", "type": "array", "item_type": "integer"},
        ]))
    else:
        console_stub.force_status(r"/api/tools/collect_evidence/schema$", 404)
    values = ["First evidence question. " * 50, "Second evidence question. " * 50, "upload/report.pdf"]
    params = {"tasks": values, "empty": [], "declared": ["retain raw"]}
    await console_stub.seed_task("parameter-shape", "collect_evidence", "2026-10-31T01:00:00+00:00",
                                 "completed", params=params)
    await page.goto(console_stub.base_url + "/task/parameter-shape")
    group = page.get_by_role("group", name="tasks", exact=True)
    await expect(group.get_by_text("3 items", exact=True)).to_be_visible()
    await expect(group.get_by_text(values[0], exact=True)).to_be_visible()
    await expect(group.get_by_text(values[1], exact=True)).to_be_visible()
    await expect(group.get_by_text(values[2], exact=True)).to_be_visible()
    more = group.get_by_role("button", name="Read more", exact=True)
    await expect(more).to_have_count(2)
    await more.first.click()
    await expect(group.get_by_role("button", name="Read less", exact=True)).to_have_attribute("aria-expanded", "true")
    assert await group.get_by_text(values[0], exact=True).inner_text() == values[0]
    await group.get_by_role("button", name="Read less", exact=True).click()
    await expect(more).to_have_count(2)
    assert await group.get_by_role("link").count() == 0
    await expect(page.get_by_role("group", name="empty", exact=True).get_by_text("No items", exact=True)).to_be_visible()
    if schema_available:
        declared = page.get_by_role("group", name="declared parameter", exact=True)
        await expect(declared.get_by_text(json.dumps(params["declared"], indent=2), exact=True)).to_be_visible()
    assert console_stub.last_call_id() is None


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("params", [{"value": value} for value in [
    12, False, None, "https://example.com/report.pdf", ["keep", 2, None],
    [["keep"]], {"exact": [1, False]}, [1, 2],
]])
async def test_parameter_ui_missing_schema_preserves_json_values(
    console_stub: SimpleNamespace, _parameter_page: Page, params: dict,
) -> None:
    """Values without declarations keep their saved text and types."""
    page = _parameter_page
    console_stub.force_status(r"/api/tools/collect_evidence/schema$", 404)
    await console_stub.seed_task("parameter-json", "collect_evidence", "2026-10-31T01:00:00+00:00",
                                 "completed", params=params)
    await page.goto(console_stub.base_url + "/task/parameter-json")
    row = page.get_by_role("group", name="value parameter", exact=True)
    value = params["value"]
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    await expect(row.get_by_text(text, exact=True)).to_be_visible()
    assert await row.get_by_role("link").count() == 0
    assert console_stub.last_call_id() is None


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("declaration,value,at_default", [
    ({"type": "string", "required": True, "default": None}, None, False),
    ({"type": "string", "required": False, "default": None}, None, True),
    ({"type": "integer", "required": False, "default": 10}, None, False),
    ({"type": "boolean", "required": False, "default": False}, False, True),
    ({"type": "integer", "required": False, "default": 0}, 0, True),
    ({"type": "string", "required": False, "default": ""}, "", True),
], ids=["required-null", "default-null", "explicit-null", "default-false", "default-zero", "default-empty"])
async def test_parameter_ui_defaults_can_be_disclosed_without_hiding_explicit_values(
    console_stub: SimpleNamespace, _parameter_page: Page, declaration: dict, value: Json, at_default: bool
) -> None:
    """Optional defaults can be shown and hidden while explicit values stay visible."""
    page = _parameter_page
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "value", **declaration}, {"name": "context", "type": "string"},
    ]))
    await console_stub.seed_task("parameter-default-contract", "google_flights", E2E_STARTED_AT, "completed",
                                params={"value": value, "context": "Stored context"})
    await page.goto(console_stub.base_url + "/task/parameter-default-contract")
    await expect(page.get_by_text("Stored context", exact=True)).to_be_visible()
    group = page.get_by_role("group", name="value parameter", exact=True, include_hidden=True)
    if at_default:
        await expect(group).to_be_hidden()
        show = page.get_by_role("button", name="1 param at default · show", exact=True)
        await expect(show).to_have_attribute("aria-expanded", "false")
        await show.focus()
        await page.keyboard.press("Enter")
        await expect(group).to_be_visible()
        hide = page.get_by_role("button", name="1 param at default · hide", exact=True)
        await expect(hide).to_have_attribute("aria-expanded", "true")
        await expect(group.get_by_text("default", exact=True)).to_be_visible()
        await hide.click()
        await expect(group).to_be_hidden()
        await show.click()
    else:
        await expect(group).to_be_visible()
        await expect(page.get_by_role("button", name=re.compile("at default"))).to_have_count(0)
    text = value if isinstance(value, str) else json.dumps(value)
    if text:
        await expect(group.get_by_text(text, exact=True)).to_be_visible()
    await expect(page.get_by_text("Stored context", exact=True)).to_be_visible()


@pytest.mark.live
@pytest.mark.timeout(60)
@pytest.mark.parametrize("value,expand", [
    ('a & "b" <c>', False),
    ("First line\n\nThird line\n" * 12, True),
], ids=["literal-text", "multiline-text"])
async def test_parameter_ui_saved_text_preserves_literal_characters_and_newlines(
    console_stub: SimpleNamespace, _parameter_page: Page, value: str, expand: bool
) -> None:
    """Saved text keeps literal characters and blank lines when read in full."""
    page = _parameter_page
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "context", "type": "string", "kind": "longtext"},
    ]))
    await console_stub.seed_task("parameter-literal-contract", "google_flights", E2E_STARTED_AT, "completed",
                                params={"context": value})
    await page.goto(console_stub.base_url + "/task/parameter-literal-contract")
    group = page.get_by_role("group", name="context parameter", exact=True)
    if expand:
        await group.get_by_role("button", name="Read more", exact=True).click()
        await expect(group.get_by_role("button", name="Read less", exact=True)).to_have_attribute("aria-expanded", "true")
    else:
        await expect(group.get_by_role("button", name="Read more", exact=True)).to_have_count(0)
    displayed = group.get_by_text(value, exact=True)
    await expect(displayed).to_be_visible()
    assert await displayed.text_content() == value
    assert await displayed.evaluate("el => el.scrollHeight <= el.clientHeight + 1")


@pytest.mark.live
@pytest.mark.timeout(60)
async def test_parameter_ui_expanded_text_survives_task_completion(
    console_stub: SimpleNamespace, _parameter_page: Page
) -> None:
    """A parameter left open stays readable after the task finishes."""
    page = _parameter_page
    value = "A long saved parameter remains open while the task completes. " * 30
    console_stub.pin_body(r"/api/tools/google_flights/schema$", _parameter_schema([
        {"name": "context", "type": "string", "kind": "longtext"},
    ]))
    await console_stub.seed_task("parameter-completion-contract", "google_flights", E2E_STARTED_AT,
                                params={"context": value})
    await page.goto(console_stub.base_url + "/task/parameter-completion-contract")
    group = page.get_by_role("group", name="context parameter", exact=True)
    await group.get_by_role("button", name="Read more", exact=True).click()
    await expect(group.get_by_role("button", name="Read less", exact=True)).to_have_attribute("aria-expanded", "true")
    await console_stub.finish("parameter-completion-contract", "completed", ended_at="2030-01-01T00:00:00+00:00")
    await expect(page.locator("#d-badge")).to_have_text("completed", timeout=_STUB_SMOKE_TIMEOUT_MS)
    await expect(group.get_by_role("button", name="Read less", exact=True)).to_have_attribute("aria-expanded", "true")
    displayed = group.get_by_text(value, exact=True)
    assert await displayed.text_content() == value
    assert await displayed.evaluate("el => el.scrollHeight <= el.clientHeight + 1")
