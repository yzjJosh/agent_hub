"""Fill the console preview with a made up dataset.

It writes the task records of the per-tool count table, 3 running, 188
completed, 22 failed, and 2 cancelled, 215 in all. It also writes the named
fixture tasks with their exact ids, start times, and params, their log and
result files, and the share layout, into the preview's DATA_DIR and
RESOURCES_DIR. The named fixtures and their child records count toward the
table, so the made up records fill in the rest.

Run this from the host. Stop the preview, wipe its folders, start it, then
seed it. The running fixtures have no live process, so a restart marks them
failed and you have to seed again. <name> is the worktree folder name, or
'main' for the main checkout.

  scripts/console_preview stop
  docker exec agent_hub \
    rm -rf /tmp/console-preview-<name>-data /tmp/console-preview-<name>-resources
  scripts/console_preview start

From a worktree, which the container has at /worktrees/<name>.

  docker exec -w /worktrees/<name> \
    -e DATA_DIR=/tmp/console-preview-<name>-data \
    -e RESOURCES_DIR=/tmp/console-preview-<name>-resources \
    -e PYTHONPATH=/worktrees/<name> \
    agent_hub python scripts/console_seed.py

From the main checkout the container has no scripts folder, so copy the
script in first. There <name> is 'main'.

  docker cp scripts/console_seed.py agent_hub:/tmp/console_seed.py
  docker exec -w /app -e PYTHONPATH=/app \
    -e DATA_DIR=/tmp/console-preview-main-data \
    -e RESOURCES_DIR=/tmp/console-preview-main-resources \
    agent_hub python /tmp/console_seed.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import struct
import uuid
import zipfile
import zlib
from datetime import datetime, timedelta, timezone

from app.lib.db import store as store_module

__all__ = []

_DEFAULT_DATA_DIR = "/tmp/console-preview-data"
_DEFAULT_RESOURCES_DIR = "/tmp/console-preview-resources"

# Counts for each tool, in the order running, completed, failed, cancelled.
_TOOLS: list[tuple[str, int, int, int, int]] = [
    ("generate_image", 1, 9, 4, 1),
    ("redfin_search", 1, 5, 2, 0),
    ("rednote_research", 1, 2, 1, 0),
    ("brave_search", 0, 46, 1, 0),
    ("github", 0, 19, 2, 0),
    ("google_flights", 0, 3, 0, 0),
    ("google_search", 0, 0, 0, 0),
    ("present_to_user", 0, 8, 0, 0),
    ("read_media_or_doc", 0, 28, 6, 1),
    ("send_or_receive_file", 0, 0, 0, 0),
    ("task", 0, 0, 0, 0),
    ("tts", 0, 12, 0, 0),
    ("web_fetch", 0, 50, 5, 0),
    ("x_search", 0, 6, 1, 0),
]

# Named fixture rows, in the order task_id, tool, status, started_age_s,
# duration_s, params, error.
_FIXTURES: list[tuple[str, str, str, int, int, dict, str | None]] = [
    ("c4a7e92f-8b34-4d2a-b7c1-5e9f0a6d3c18", "rednote_research", "running", 161, 0,
     {"query": "杭州咖啡", "count": 5, "raw_notes": False}, None),
    ("9c1d4e07-b2f6-4a8d-9e31-7a5c0d8f2b64", "rednote_research", "failed", 7346, 83,
     {"query": "上海周末去哪玩", "count": 3}, "Research failed: no analysis results gathered."),
    ("f3a1c9d2-5e08-4c77-a1b3-9d24f8e6c051", "rednote_research", "completed", 19765, 348,
     {"query": "成都火锅", "count": 5}, None),
    ("8b2e47aa-91c3-4f60-b7d2-4a8e0c19f5b7", "rednote_research", "completed", 33122, 339,
     {"query": "北京胡同漫步", "count": 4}, None),
    ("0797d2e6-3c41-4f8a-9b27-6d8e1a0f5c42", "web_fetch", "completed", 4021, 4,
     {"url": "https://www.xiaohongshu.com/search_result?keyword=%E6%9D%AD%E5%B7%9E%E5%92%96%E5%95%A1"}, None),
    ("media-run-1", "generate_image", "running", 372, 0,
     {"prompt": "A harbor at dusk seen from a drone, cinematic", "aspect_ratio": "16:9"}, None),
    # A finished run whose result text names two share files. gen_001.png is
    # in the seeded share and gen_999.png is not.
    ("media-result-1", "generate_image", "completed", 7190, 114,
     {"prompt": "Watercolor fox on a rainy window seat", "aspect_ratio": "1:1"}, None),
    ("redfin-run-1", "redfin_search", "running", 41, 0,
     {"location": "94103", "limit": 10, "sold": False}, None),
    # Finished runs whose result body is a different kind of file each time.
    ("wf-code-1", "web_fetch", "completed", 9001, 3, {"url": "https://example.com/add.py"}, None),
    ("wf-text-1", "web_fetch", "completed", 9002, 2, {"url": "https://example.com/notes.txt"}, None),
    ("wf-csv-1", "web_fetch", "completed", 9003, 2, {"url": "https://example.com/sales.csv"}, None),
    ("wf-json-1", "web_fetch", "completed", 9004, 2, {"url": "https://example.com/config.json"}, None),
    ("wf-html-1", "web_fetch", "completed", 9005, 2, {"url": "https://example.com/index.html"}, None),
    ("wf-image-1", "web_fetch", "completed", 9006, 2, {"url": "https://example.com/photo.png"}, None),
    ("wf-video-1", "web_fetch", "completed", 9007, 3, {"url": "https://example.com/clip.mp4"}, None),
    ("wf-audio-1", "web_fetch", "completed", 9008, 2, {"url": "https://example.com/track.wav"}, None),
    ("wf-pdf-1", "web_fetch", "completed", 9009, 3, {"url": "https://example.com/spec.pdf"}, None),
    ("wf-xlsx-1", "web_fetch", "completed", 9010, 4, {"url": "https://example.com/report.xlsx"}, None),
    ("wf-docx-1", "web_fetch", "completed", 9011, 4, {"url": "https://example.com/memo.docx"}, None),
    ("wf-zip-1", "web_fetch", "completed", 9012, 2, {"url": "https://example.com/bundle.zip"}, None),
    ("wf-empty-1", "web_fetch", "completed", 9013, 1, {"url": "https://example.com/empty"}, None),
    ("wf-file-1", "web_fetch", "completed", 9014, 2, {"url": "https://example.com/blob.bin"}, None),
]

# Subtasks of the running rednote fixture. Both are finished so the running
# counts still add up.
_CHILD_WEB_FETCH = ("child-webfetch", "web_fetch", "completed", 155, 2,
                    {"url": "https://www.xiaohongshu.com/explore/66f2b8a1000000001c03a9f2"}, None)
_CHILD_READ = ("child-readmedia", "read_media_or_doc", "completed", 98, 5,
               {"resource_path": "downloads/note_img1.png"}, None)

_PARENT = "c4a7e92f-8b34-4d2a-b7c1-5e9f0a6d3c18"

_NAMED = (*_FIXTURES, _CHILD_WEB_FETCH, _CHILD_READ)

_STATUS_KEYS = ("running", "completed", "failed", "cancelled")

# Params given to the made up records of each tool.
_PARAM_TEMPLATES: dict[str, list[dict]] = {
    "generate_image": [{"prompt": "Isometric illustration of a tea room", "aspect_ratio": "1:1"},
                       {"prompt": "Macro shot of matcha being whisked", "aspect_ratio": "4:3", "upscale": True}],
    "redfin_search": [{"location": "94103", "limit": 10, "sold": False},
                      {"location": "Palo Alto", "limit": 20, "sold": True}],
    "brave_search": [{"query": "coffee roasters near me"}, {"query": "bay area hiking trails"}],
    "github": [{"action": "search_repos", "query": "agent"}, {"action": "list_releases", "repo": "jlowin/fastmcp"}],
    "rednote_research": [{"query": "咖啡探店", "count": 5}, {"query": "露营装备清单", "count": 3}],
    "google_flights": [{"origin": "SFO", "destination": "JFK", "depart_date": "2026-09-01"},
                       {"origin": "LAX", "destination": "HNL", "depart_date": "2026-10-12"}],
    "google_search": [{"query": "weather"}],
    "present_to_user": [{"content": "summary of the latest run"}],
    "read_media_or_doc": [{"resource_path": "downloads/capture.png"}, {"resource_path": "tts/tts_20260820_171913.opus"}],
    "send_or_receive_file": [{}],
    "task": [{"action": "status", "task_id": "c4a7e92f"}, {"action": "logs", "task_id": "9c1d4e07"}],
    "tts": [{"text": "Here is the latest status update", "format": "opus"}],
    "web_fetch": [{"url": "https://example.com/page/1"}, {"url": "https://example.com/page/2"}],
    "x_search": [{"query": "agent console"}],
}

_FAILED_ERRORS = [
    "Browser launch failed: executable doesn't exist at /tmp/chrome",
    "Research failed: no analysis results gathered.",
    "Timeout 60000ms exceeded while waiting for navigation",
    "LLM backend error: 502 Bad Gateway",
]

_RESULT_BYTES_0797 = 793191  # the completed web_fetch fixture's exact size
_LINE_WIDTH = 78
_SPREAD_DAYS = 26
_RUN_LINES_238 = 238  # log lines the running fixture gets
_FAILED_LOG_LINES = 250  # log lines the failed fixture gets

_RNG_SEED = 20260823


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Make a small PNG of one solid color."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def _zip_with(entry: str, content: bytes) -> bytes:
    """Make a zip file holding one file with the given name."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(entry, content)
    return buf.getvalue()


def _fill_up_to(path: str, size: int, chunk_text: str, rng: random.Random) -> None:
    """Write text to a file until the file is exactly the given size."""
    chunk = chunk_text.encode("utf-8")
    out = bytearray()
    first = True
    while len(out) < size:
        prefix = chunk if first else b""
        first = False
        fill = min(_LINE_WIDTH, max(size - len(out) - len(prefix) - 1, 0))
        out.extend(prefix)
        out.extend("".join(rng.choices("abcdefghijklmnopqrstuvwxyz-", k=fill)).encode("ascii"))
        out.extend(b"\n")
    with open(path, "wb") as fh:
        fh.write(bytes(out[:size]))


def _run_log_lines(rng: random.Random) -> list[str]:
    """Make the log lines for the running fixture."""
    lines = [
        "[BrowserManager] Connecting to CDP: http://localhost:9222",
        "[BrowserManager] Download dir: /dev/shm/agent/downloads",
        "[BrowserManager.new_page] Created page: 3f9c81d2a7e04b5c8d6f2a1b9e0c7d43",
        "[BrowserManager.navigate] Navigating to https://www.xiaohongshu.com (wait_until=domcontentloaded)",
        "[BrowserManager.navigate] Success",
        "[BrowserManager.navigate] Navigating to https://www.xiaohongshu.com/search_result?keyword=%E6%9D%AD%E5%B7%9E%E5%92%96%E5%95%A1",
        "[BrowserManager.wait_for_load] Waiting for state: domcontentloaded",
        "[BrowserManager.wait_for_load] Success",
        "[BrowserManager.snapshot] Capturing snapshot...",
        "[BrowserManager.snapshot] Warning: networkidle wait failed: Timeout 5000ms exceeded.",
        "[BrowserManager.snapshot] HTML length: 842113, URL: https://www.xiaohongshu.com/search_result?keyword=%E6%9D%AD%E5%B7%9E%E5%92%96%E5%95%A1",
        "[RednoteSearch] Found 5 notes in initial results",
        "[RednoteSearch] Opening note 1/5: /explore/66f2b8a1000000001c03a9f2",
        "[RednoteSearch] Note 1: 4 body images captured",
        "[ImagePipeline] Downloading image 1/4: note_66f2b8a1_img1.jpg (182 KB)",
        "[ImagePipeline] Downloading image 2/4: note_66f2b8a1_img2.jpg (203 KB)",
    ]
    note = 1
    img = 3
    while len(lines) < _RUN_LINES_238:
        r = rng.random()
        if r < 0.30:
            img = img % 4 + 1
            lines.append(f"[ImagePipeline] Downloading image {img}/4: note_66f2b8a1_img{img}.jpg ({rng.randint(90, 420)} KB)")
        elif r < 0.55:
            lines.append(f"[ImagePipeline] OCR: note_66f2b8a1_img{img}.jpg ({rng.randint(120, 900)} ms)")
        elif r < 0.70:
            lines.append(f"[Vision.describe] Describing note_66f2b8a1_img{img}.jpg (backend: local)")
        elif r < 0.80:
            lines.append(f"[Vision.describe] OK: cupping scene, {rng.randint(2, 6)} details extracted")
        elif r < 0.88:
            note = note % 5 + 1
            lines.append(f"[RednoteSearch] Opening note {note}/5: /explore/66e9c3d4000000001e07b2c8")
        elif r < 0.95:
            lines.append(f"[LLM.analyze] Analyzing note {note}/5 (model: reason backend, {rng.randint(4000, 14000)} tokens context)")
        else:
            lines.append(f"[LLM.analyze] Note {note} analyzed in {rng.randint(2, 14)}.{rng.randint(0, 9)}s")
    # A block of browser log output near the end.
    i = len(lines) - 12
    lines[i:i] = [
        "=========================== logs ===========================",
        "[Browser.page.screenshot] Capturing page screenshot...",
        "Warning: Retry 1/3 — tab crashed, respawning",
        "Error: page.evaluate: Execution context was destroyed",
        "============================ logs end ===========================",
    ]
    return lines[:_RUN_LINES_238]


def _failed_log_lines(rng: random.Random) -> list[str]:
    """Make the log lines for the failed fixture."""
    lines = [
        "[BrowserManager] Connecting to CDP: http://localhost:9222",
        "[BrowserManager] Download dir: /dev/shm/agent/downloads",
        "[BrowserManager.new_page] Created page: 9f0c2d41a6e84b7c3f5a8d0e2b1c4f9a",
        "[BrowserManager.navigate] Navigating to https://www.xiaohongshu.com (wait_until=domcontentloaded)",
        "[BrowserManager.navigate] Success",
        "[BrowserManager.navigate] Navigating to https://www.xiaohongshu.com/search_result?keyword=%E4%B8%8A%E6%B5%B7%E5%91%A8%E6%9C%AB%E5%8E%BB%E5%93%AA%E7%8E%A9",
        "[BrowserManager.wait_for_load] Waiting for state: domcontentloaded",
        "[BrowserManager.wait_for_load] Success",
        "[BrowserManager.snapshot] Capturing snapshot...",
        "[BrowserManager.snapshot] HTML length: 512004, URL: https://www.xiaohongshu.com/search_result?keyword=…",
        "[RednoteSearch] Found 3 notes in initial results",
    ]
    note = 1
    img = 1
    tail = 5
    while len(lines) < _FAILED_LOG_LINES - tail:
        r = rng.random()
        if r < 0.28:
            img = img % 2 + 1
            lines.append(f"[ImagePipeline] Downloading image {img}/2: note_66aa12f4_img{img}.jpg ({rng.randint(90, 420)} KB)")
        elif r < 0.50:
            lines.append(f"[BrowserManager.snapshot] Warning: networkidle wait failed: Timeout {rng.randint(3000, 8000)}ms exceeded.")
        elif r < 0.68:
            note = note % 3 + 1
            lines.append(f"[RednoteSearch] Opening note {note}/3: /explore/66aa12f4000000001e01a3b8")
        elif r < 0.86:
            lines.append(f"[ImagePipeline] OCR: note_66aa12f4_img{img}.jpg ({rng.randint(120, 900)} ms)")
        else:
            lines.append(f"[LLM.analyze] Analyzing note {note}/3 (model: reason backend, {rng.randint(4000, 14000)} tokens context)")
    lines.extend([
        "[RednoteSearch] Note 1: 0 body images captured",
        "[RednoteSearch] Note 2: 0 body images captured",
        "[RednoteSearch] Note 3: 0 body images captured",
        "[RednoteSearch] No notes produced analysis output",
        "Error: Research failed: no analysis results gathered.",
    ])
    return lines[:_FAILED_LOG_LINES]


def _named_counts() -> dict[str, dict[str, int]]:
    """Count the named fixture and child records for each tool and status."""
    counts: dict[str, dict[str, int]] = {}
    for _tid, tool, status, _age, _dur, _params, _error in _NAMED:
        per_tool = counts.setdefault(tool, {key: 0 for key in _STATUS_KEYS})
        per_tool[status] += 1
    return counts


async def _insert_task(now: datetime, task_id: str, tool: str, status: str,
                       age_s: int, dur_s: int, params: dict, error: str | None,
                       parent: str | None) -> None:
    started = now - timedelta(seconds=age_s)
    ended = started + timedelta(seconds=dur_s)
    await store_module.task_store.insert_running(
        task_id, parent, started.isoformat(), f"runs/{task_id}")
    await store_module.tool_store.insert(task_id, tool, None, params)
    if status != "running":
        await store_module.task_store.finish(task_id, status, error=error, ended_at=ended.isoformat())


async def _seed_records(data_dir: str, now: datetime) -> None:
    rng = random.Random(_RNG_SEED)
    record_store = store_module.init_store(os.path.join(data_dir, "records.db"))
    await record_store.open()
    named = _named_counts()
    for tool, running, completed, failed, cancelled in _TOOLS:
        per_tool_named = named.get(tool, {key: 0 for key in _STATUS_KEYS})
        n = 0
        for status, total in (("running", running), ("completed", completed),
                              ("failed", failed), ("cancelled", cancelled)):
            for _ in range(total - per_tool_named[status]):
                n += 1
                tid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"console-seed/{tool}/{status}/{n}"))
                age = 240 + rng.randint(0, _SPREAD_DAYS * 86400)
                dur = rng.randint(1, 400)
                error = None
                if status == "failed":
                    error = rng.choice(_FAILED_ERRORS)
                elif status == "cancelled":
                    error = "Cancelled by user."
                params = rng.choice(_PARAM_TEMPLATES[tool])
                await _insert_task(now, tid, tool, status, age, dur, dict(params), error, None)
    for tid, tool, status, age, dur, params, error in _NAMED:
        parent = _PARENT if tid in ("child-webfetch", "child-readmedia") else None
        await _insert_task(now, tid, tool, status, age, dur, dict(params), error, parent)
    await record_store.close()


def _runs_dir(data_dir: str, task_id: str) -> str:
    path = os.path.join(data_dir, "runs", task_id)
    os.makedirs(path, exist_ok=True)
    return path


def _seed_run_files(data_dir: str, rng: random.Random) -> None:
    with open(os.path.join(_runs_dir(data_dir, "c4a7e92f-8b34-4d2a-b7c1-5e9f0a6d3c18"), "log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_run_log_lines(rng)) + "\n")
    with open(os.path.join(_runs_dir(data_dir, "9c1d4e07-b2f6-4a8d-9e31-7a5c0d8f2b64"), "log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_failed_log_lines(rng)) + "\n")
    # The finished web_fetch fixture gets an empty log and a result of an exact size.
    with open(os.path.join(_runs_dir(data_dir, "0797d2e6-3c41-4f8a-9b27-6d8e1a0f5c42"), "log"), "w", encoding="utf-8") as fh:
        fh.write("")
    _fill_up_to(os.path.join(_runs_dir(data_dir, "0797d2e6-3c41-4f8a-9b27-6d8e1a0f5c42"), "result"),
                _RESULT_BYTES_0797, "# Xiaohongshu — 杭州咖啡", rng)
    for tid, body in (
        ("f3a1c9d2-5e08-4c77-a1b3-9d24f8e6c051", "# 成都火锅 — top 5 notes\n\n1. 蜀大侠火锅（春熙路店）\n2. 小龙坎老火锅\n3. 大龙燚\n4. 周师兄大刀腰片\n5. 巴蜀大宅门\n"),
        ("8b2e47aa-91c3-4f60-b7d2-4a8e0c19f5b7", "# 北京胡同漫步 — top 4 notes\n\n1. 南锣鼓巷\n2. 五道营胡同\n3. 烟袋斜街\n4. 后海 — 什刹海\n"),
        ("media-result-1",
         "Saved: media_gen/gen_001.png\n"
         "WebDAV download:\n"
         'curl -o gen_001.png "http://127.0.0.1:8315/dav/media_gen/gen_001.png"\n'
         "Saved: media_gen/gen_999.png\n"
         "WebDAV download:\n"
         'curl -o gen_999.png "http://127.0.0.1:8315/dav/media_gen/gen_999.png"\n'),
    ):
        with open(os.path.join(_runs_dir(data_dir, tid), "result"), "w", encoding="utf-8") as fh:
            fh.write(body)
    # Result bodies, one for each kind of file the preview can show.
    _kind_bodies = {
        "wf-code-1": b"def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n",
        "wf-text-1": b"Plain text result body.\nSecond line.\nThird line.\n",
        "wf-csv-1": b"name,qty,price\napple,3,1.20\nbanana,5,0.50\ncherry,2,4.00\n",
        "wf-json-1": b'{"name": "agent", "version": 3, "ok": true}\n',
        "wf-html-1": b"<!DOCTYPE html>\n<html><head><title>Page</title></head>\n"
                     b"<body><h1>Page</h1><p>Body</p></body></html>\n",
        "wf-image-1": _png(160, 100, (0x8F, 0xA3, 0x89)),
        "wf-video-1": b"\x00\x00\x00\x18ftypisom" + b"\x00" * 96,
        "wf-audio-1": b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32,
        "wf-pdf-1": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n",
        "wf-xlsx-1": _zip_with("xl/workbook.xml", b"<workbook/>"),
        "wf-docx-1": _zip_with("word/document.xml", b"<document/>"),
        "wf-zip-1": _zip_with("notes.txt", b"seeded archive entry\n"),
        "wf-empty-1": b"",
        "wf-file-1": b"\x00\x01\x02\xff\xfe\x00\x01\x02\x03\x04\x05",
    }
    for tid, body in _kind_bodies.items():
        with open(os.path.join(_runs_dir(data_dir, tid), "result"), "wb") as fh:
            fh.write(body)


# Files to put in the share, with the size each one gets.
_SHARE_FILES: list[tuple[str, int]] = [
    ("downloads/1beebf7c2c7c4d42adf31a28d483affe.docx", 36864),
    ("downloads/9a2f83dae22a42a28cd6b1f975b4dec0.png", 200),
    ("downloads/d0f9a955df4b45b39bfba8e96a1777f2.png", 200),
    ("downloads/downloaded_file", 11264),
    ("downloads/61df8b55bc9349e4be424a5fb8bf3a7f.png", 128),
    ("downloads/707c9df739a24cf283c2ef77d510c8cf.png", 128),
    ("downloads/743cc58f8e1e4469a8cf24885ac80e5f.png", 128),
    ("downloads/9b801927f8b0408482ae45db543c5c3e.png", 128),
    ("downloads/a1123a41fd3a47b9ad78f6f7b17c81b1.png", 128),
    ("downloads/a667581889a24a2abc94c155ef0439ea.png", 128),
    ("downloads/bd4e1f0172b04fe9847f73c66123a6a3.png", 128),
    ("downloads/f472c837e2f84113a3f6a9080d5fded7.png", 128),
    ("tts/tts_20260819_184537.opus", 43008),
    ("tts/tts_20260819_184731.opus", 47104),
    ("tts/tts_20260819_185812.opus", 109312),
    ("tts/tts_20260819_195919.opus", 120576),
    ("tts/tts_20260820_171913.opus", 752640),
    ("github/redfin-94103.json", 700000),
    ("github/pr-142-diff.txt", 600000),
]

# Folders to fill, with how many files each one gets.
_BULK_FOLDERS: list[tuple[str, int]] = [
    ("debug_flake", 108),
    ("media_gen", 78),
    ("upload", 23),
]

_PNG_PALETTE: list[tuple[int, int, int]] = [
    (0xC4, 0xA6, 0x7F), (0x8F, 0xA3, 0x89), (0x94, 0xA3, 0xBF), (0xC0, 0x8F, 0x83),
    (0xA7, 0x9C, 0xC0), (0x83, 0xA3, 0xAA), (0xB8, 0xA8, 0x7F), (0xA3, 0x8F, 0xA8),
]


def _seed_share(resources_dir: str, rng: random.Random) -> None:
    os.makedirs(resources_dir, exist_ok=True)
    for rel, size in _SHARE_FILES:
        path = os.path.join(resources_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        name = os.path.basename(rel)
        if name.endswith(".png"):
            data = _png(64, 40, _PNG_PALETTE[rng.randrange(len(_PNG_PALETTE))])
            with open(path, "wb") as fh:
                fh.write(data if size <= len(data) else data + b"\x00" * (size - len(data)))
        elif name == "downloaded_file":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("placeholder text file seeded by console_seed.py\n" * 100)
        else:
            _fill_up_to(path, size, "sample", rng)
    for folder, count in _BULK_FOLDERS:
        os.makedirs(os.path.join(resources_dir, folder), exist_ok=True)
        for i in range(count):
            if folder == "debug_flake":
                if i % 6 == 0:
                    rel = f"debug_flake/trace_{i:03d}.log"
                    _fill_up_to(os.path.join(resources_dir, rel), 200000 + i * 137, "log", rng)
                else:
                    rel = f"debug_flake/capture_{i:03d}.png"
                    with open(os.path.join(resources_dir, rel), "wb") as fh:
                        fh.write(_png(64, 40, _PNG_PALETTE[i % len(_PNG_PALETTE)]))
            elif folder == "media_gen":
                if i % 5 == 0:
                    rel = f"media_gen/gen_{i:03d}.meta.json"
                    _fill_up_to(os.path.join(resources_dir, rel), 2000 + i * 31, "meta", rng)
                else:
                    rel = f"media_gen/gen_{i:03d}.png"
                    with open(os.path.join(resources_dir, rel), "wb") as fh:
                        fh.write(_png(64, 40, _PNG_PALETTE[i % len(_PNG_PALETTE)]))
            else:
                if i < 3:
                    rel = f"upload/notes_{i}.md"
                    with open(os.path.join(resources_dir, rel), "w", encoding="utf-8") as fh:
                        fh.write("# upload note\n\nseeded markdown content\n" * 40)
                else:
                    rel = f"upload/upload_{i:03d}.png"
                    with open(os.path.join(resources_dir, rel), "wb") as fh:
                        fh.write(_png(64, 40, _PNG_PALETTE[i % len(_PNG_PALETTE)]))


def _seed_sidecars(data_dir: str, resources_dir: str) -> None:
    """Write the files sidecar of the finished generate_image fixture.

    It saved media_gen/gen_001.png, which is in the seeded share, and
    media_gen/gen_999.png, which a later garbage collection has removed.
    """
    st = os.stat(os.path.join(resources_dir, "media_gen", "gen_001.png"))
    lines = (
        json.dumps({"path": "media_gen/gen_001.png", "size": st.st_size, "mtime": int(st.st_mtime)}),
        json.dumps({"path": "media_gen/gen_999.png", "size": 234567, "mtime": int(st.st_mtime) - 120}),
    )
    with open(os.path.join(_runs_dir(data_dir, "media-result-1"), "files"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


async def main() -> None:
    """Fill DATA_DIR and RESOURCES_DIR with the made up dataset.

    They fall back to the preview folders when they are not set.
    """
    data_dir = os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR)
    resources_dir = os.environ.get("RESOURCES_DIR", _DEFAULT_RESOURCES_DIR)
    now = datetime.now(timezone.utc)
    try:
        await _seed_records(data_dir, now)
        await asyncio.to_thread(_seed_run_files, data_dir, random.Random(_RNG_SEED + 2))
        await asyncio.to_thread(_seed_share, resources_dir, random.Random(_RNG_SEED + 3))
        await asyncio.to_thread(_seed_sidecars, data_dir, resources_dir)
    finally:
        await store_module.task_store.close()
    total = sum(running + completed + failed + cancelled for _, running, completed, failed, cancelled in _TOOLS)
    print(f"seeded {total} records into {data_dir}")


if __name__ == "__main__":
    asyncio.run(main())
