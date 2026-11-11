"""Makes thumbnails for files.

Each job runs on its own. A job that fails does not stop later jobs.
"""

from __future__ import annotations

import asyncio
import os
import threading

from PIL import Image

from app.lib.browser.browser_manager import BrowserManager
from app.lib.cli.cli_runner import run_cli
from app.lib.shared import config
from app.lib.shared.paths import resolve_path_sync, resources_dir
from app.lib.render import cache, gen_state, office
from app.lib.render.classify import RenderKind, classify

__all__ = [
    "start_dir_thumbnails",
    "start_thumbnail_job",
    "stop_thumbnails",
    "thumb_engine",
]

_ENQUEUED: set[str] = set()
_LIVE_JOBS: set[asyncio.Task] = set()
_browser_lock: asyncio.Lock | None = None
_browser_lock_loop: asyncio.AbstractEventLoop | None = None
_render_sem: asyncio.Semaphore | None = None
_render_sem_loop: asyncio.AbstractEventLoop | None = None
# How many renders may run at once, behind one shared guard.
_RENDER_CAP = 4
# Bind hosts that mean every address. The server is reached on loopback then.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_LOOPBACK_HOST = "127.0.0.1"
_CDP_TIMEOUT_MS = 15000
_THUMB_W, _THUMB_H = 1024, 640


def thumb_engine(kind: RenderKind) -> str | None:
    """Say what makes the thumbnail for a kind of file.

    Args:
        kind: What sort of file this is.

    Returns:
        The name of what makes the thumbnail, or None when this kind gets no
        thumbnail.
    """
    if kind is RenderKind.IMAGE:
        return "image"
    if kind is RenderKind.VIDEO:
        return "video"
    if kind is RenderKind.PDF:
        return "pdf"
    if kind is RenderKind.OFFICE_PPT:
        return "ppt_pdf"
    if kind in (RenderKind.MARKDOWN, RenderKind.CODE, RenderKind.CSV, RenderKind.JSON,
                RenderKind.JSONL, RenderKind.YAML, RenderKind.XML, RenderKind.HTML,
                RenderKind.OFFICE_DOC, RenderKind.XLSX, RenderKind.XLS,
                RenderKind.TEXT):
        return "doc"
    return None


def start_thumbnail_job(key: str, kind: RenderKind, path: str) -> bool:
    """Start a thumbnail job for one file.

    Call this on the event loop only.

    Nothing happens when the file is already in the pipeline or is being
    worked on, or when it gets no thumbnail.

    Args:
        key: The name the file is saved under.
        kind: What sort of file this is.
        path: A path inside the shared folder.

    Returns:
        True when a job was started.
    """
    if key in _ENQUEUED or key in gen_state.generating_keys():
        return False
    if thumb_engine(kind) is None:
        return False
    _ENQUEUED.add(key)
    gen_state.gen_add(key)
    task = asyncio.get_running_loop().create_task(_thumb_job(key, kind, path), name=f"thumb-{key[:8]}")
    _LIVE_JOBS.add(task)
    task.add_done_callback(_LIVE_JOBS.discard)
    return True


def _files_in_dir(rel_dir: str) -> list[tuple[str, str, int]]:
    """Find the files in one folder.

    Args:
        rel_dir: A folder inside the shared folder.

    Returns:
        The full path, the shared path, and the change time of each file.

    Raises:
        FileNotFoundError: If there is no such folder.
    """
    base = os.path.realpath(resources_dir())
    target = base if not rel_dir else resolve_path_sync(base, rel_dir)
    if target is None or not os.path.isdir(target):
        raise FileNotFoundError(rel_dir)
    found: list[tuple[str, str, int]] = []
    with os.scandir(target) as it:
        for entry in it:
            if not entry.is_file():
                continue
            # The path is relative to the shared folder, never a bare file name.
            rel = os.path.relpath(entry.path, base).replace(os.sep, "/")
            found.append((entry.path, rel, int(entry.stat().st_mtime)))
    return found


async def _scan_dir_for_thumbs(rel_dir: str) -> list[tuple[str, RenderKind, str]]:
    """Find the files in one folder that still need a thumbnail.

    Args:
        rel_dir: A folder inside the shared folder.

    Returns:
        The saved name, kind, and path of each file that needs one.

    Raises:
        FileNotFoundError: If there is no such folder.
    """
    jobs: list[tuple[str, RenderKind, str]] = []
    for full, rel, mtime in await asyncio.to_thread(_files_in_dir, rel_dir):
        kind = await classify(full)
        if thumb_engine(kind) is None:
            continue
        key = cache.key_for(rel, mtime)
        if await cache.thumb_status(key) in ("pending", "failed"):
            jobs.append((key, kind, rel))
    return jobs


async def start_dir_thumbnails(rel_dir: str) -> int:
    """Start a thumbnail job for every file in one folder that needs one.

    Args:
        rel_dir: A folder inside the shared folder.

    Returns:
        How many jobs were started.
    """
    jobs = await _scan_dir_for_thumbs(rel_dir)
    started = 0
    for key, kind, rel in jobs:
        if start_thumbnail_job(key, kind, rel):
            started += 1
    return started


def _resolve_existing_file(path: str) -> str | None:
    """Find where a shared file is on disk.

    Args:
        path: A path inside the shared folder.

    Returns:
        The full path, or None when the path is not allowed or there is no
        such file.
    """
    full = resolve_path_sync(resources_dir(), path)
    if full is None or not os.path.isfile(full):
        return None
    return full


async def _thumb_job(key: str, kind: RenderKind, path: str) -> None:
    """Make one thumbnail: get the input ready, then render behind the cap.

    The input is made ready first without holding a render slot, so a slow
    wait here does not block other renders. A failure in either stage leaves
    a failure mark. Both in-flight entries are released when the job ends.

    Args:
        key: The name the file is saved under.
        kind: What sort of file this is.
        path: A path inside the shared folder.
    """
    try:
        full = await asyncio.to_thread(_resolve_existing_file, path)
        if full is None:
            raise FileNotFoundError(path)
        engine = thumb_engine(kind)
        if engine in ("doc", "ppt_pdf"):
            # The prepare stage may wait for a conversion without a render slot.
            if await office.ensure_doc_for_thumb(kind, key, path, full) != "published":
                raise RuntimeError("document unavailable")
        async with _render_semaphore():
            if engine == "image":
                await _thumb_image(full, key)
            elif engine == "video":
                await _thumb_video(full, key)
            elif engine == "pdf":
                await _thumb_pdf(full, key)
            elif engine == "ppt_pdf":
                await _thumb_pdf(os.path.join(cache.render_root(), key, cache.OFFICE_PDF_ASSET), key)
            elif engine == "doc":
                await _render_doc_page(key)
            else:
                raise RuntimeError(f"no engine for {kind}")
    except Exception as exc:
        await cache.mark_failed(key, "thumb", (str(exc) or "thumbnail failed")[:200])
    finally:
        gen_state.gen_remove(key)
        _ENQUEUED.discard(key)


async def stop_thumbnails() -> None:
    """Stop thumbnail making at shutdown.

    There is no persistent browser connection to close. Each render opens
    and tears down its own. The dispatch state is cleared. In-flight jobs
    are left to the loop's own shutdown, which cancels them and runs their
    cleanup.
    """
    _ENQUEUED.clear()
    _LIVE_JOBS.clear()


def _render_semaphore() -> asyncio.Semaphore:
    """Get the guard that bounds how many renders run at once.

    Returns:
        The guard.
    """
    global _render_sem, _render_sem_loop
    loop = asyncio.get_running_loop()
    if _render_sem is None or _render_sem_loop is not loop:
        _render_sem = asyncio.Semaphore(_RENDER_CAP)
        _render_sem_loop = loop
    return _render_sem


def _browser_guard() -> asyncio.Lock:
    """Get the lock that serializes the browser connect and teardown.

    Returns:
        The lock.
    """
    global _browser_lock, _browser_lock_loop
    loop = asyncio.get_running_loop()
    if _browser_lock is None or _browser_lock_loop is not loop:
        _browser_lock = asyncio.Lock()
        _browser_lock_loop = loop
    return _browser_lock


def _view_base() -> str:
    """Get the address the browser uses to reach this server.

    Returns:
        The base address, with no trailing slash.
    """
    host, port = config.http_bind()
    if host in _WILDCARD_HOSTS:
        host = _LOOPBACK_HOST
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _cover_crop(box: tuple[int, int], ratio_w: int = 16, ratio_h: int = 10) -> tuple[int, int, int, int]:
    w, h = box
    target = ratio_w / ratio_h
    if w / h > target:
        new_w = int(h * target)
        left = (w - new_w) // 2
        return left, 0, left + new_w, h
    new_h = int(w / target)
    top = (h - new_h) // 2
    return 0, top, w, top + new_h


def _atomic_save_png(im: Image.Image, key: str) -> None:
    key_dir = os.path.join(cache.render_root(), key)
    os.makedirs(key_dir, exist_ok=True)
    staging = os.path.join(key_dir, f".thumb-{os.getpid()}-{threading.get_ident()}.png")
    im.save(staging, "PNG")
    os.replace(staging, os.path.join(key_dir, "thumb.png"))


def _resize_crop(im: Image.Image) -> Image.Image:
    box = _cover_crop(im.size)
    return im.crop(box).resize((_THUMB_W, _THUMB_H))


async def _thumb_image(full_path: str, key: str) -> None:
    """Make the thumbnail for an image.

    Args:
        full_path: Where the image is on disk.
        key: The name the file is saved under.
    """

    def _run() -> None:
        with Image.open(full_path) as im:
            im.load()
            _atomic_save_png(_resize_crop(im), key)

    await asyncio.to_thread(_run)


async def _thumb_video(full_path: str, key: str) -> None:
    """Make the thumbnail for a video.

    The picture comes from near the start of the video.

    Args:
        full_path: Where the video is on disk.
        key: The name the file is saved under.

    Raises:
        RuntimeError: If no picture can be pulled out of the video.
    """
    probe = await run_cli(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", full_path],
        "ffprobe", timeout=10,
    )
    seek = 0.0
    if probe.returncode == 0 and probe.stdout:
        try:
            seek = min(float(probe.stdout) * 0.1, 5.0)
        except ValueError:
            pass
    key_dir = os.path.join(cache.render_root(), key)
    await asyncio.to_thread(os.makedirs, key_dir, exist_ok=True)
    frame = os.path.join(key_dir, ".frame.png")
    result = await run_cli(
        ["ffmpeg", "-y", "-ss", f"{seek}", "-i", full_path, "-frames:v", "1", frame],
        "ffmpeg", timeout=30,
    )
    if result.returncode != 0 or not await asyncio.to_thread(os.path.exists, frame):
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")

    def _run() -> None:
        with Image.open(frame) as im:
            im.load()
            _atomic_save_png(_resize_crop(im), key)
        os.remove(frame)

    await asyncio.to_thread(_run)


def _page_files(key_dir: str) -> list[str]:
    """List the page pictures made from a PDF.

    Args:
        key_dir: The folder that holds one file's saved bits.

    Returns:
        The names of the page pictures, in order.
    """
    return sorted(f for f in os.listdir(key_dir) if f.startswith(".page-"))


async def _thumb_pdf(full_path: str, key: str) -> None:
    """Make the thumbnail for a PDF.

    The picture comes from the first page.

    Args:
        full_path: Where the PDF is on disk.
        key: The name the file is saved under.

    Raises:
        RuntimeError: If the first page cannot be drawn.
    """
    key_dir = os.path.join(cache.render_root(), key)
    await asyncio.to_thread(os.makedirs, key_dir, exist_ok=True)
    prefix = os.path.join(key_dir, ".page")
    result = await run_cli(
        ["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1", full_path, prefix],
        "pdftoppm", timeout=30,
    )
    pages = await asyncio.to_thread(_page_files, key_dir)
    if result.returncode != 0 or not pages:
        raise RuntimeError(f"pdftoppm failed: {result.stderr[-200:]}")
    page_path = os.path.join(key_dir, pages[0])

    def _run() -> None:
        with Image.open(page_path) as im:
            im.load()
            _atomic_save_png(_resize_crop(im), key)
        os.remove(page_path)

    await asyncio.to_thread(_run)


async def _render_doc_page(key: str) -> None:
    """Shoot the saved page and save it as the thumbnail.

    The browser is connected to and let go of for every render.

    Args:
        key: The name the file is saved under.

    Raises:
        RuntimeError: If the page cannot be opened or shot.
    """
    async with _browser_guard():
        manager = await BrowserManager.create()
        target_id: str | None = None
        try:
            target_id = await manager.new_page()
            ok = await manager.set_viewport_size(target_id, _THUMB_W, _THUMB_H)
            if not ok:
                raise RuntimeError("thumbnail viewport failed")
            ok = await manager.navigate(target_id, f"{_view_base()}/view/doc/{key}?thumb=1", wait_until="load", timeout=_CDP_TIMEOUT_MS)
            # The saved page can go away while the job is running.
            if not ok:
                raise RuntimeError("thumbnail page did not load")
            key_dir = os.path.join(cache.render_root(), key)
            staging = os.path.join(key_dir, f".cdp-{os.getpid()}.jpg")
            await asyncio.to_thread(os.makedirs, key_dir, exist_ok=True)
            saved = await manager.screenshot(target_id, staging, full_page=False)
            if saved is None:
                raise RuntimeError("thumbnail screenshot failed")

            def _run() -> None:
                with Image.open(staging) as im:
                    im.load()
                    _atomic_save_png(_resize_crop(im), key)
                os.remove(staging)

            await asyncio.to_thread(_run)
        finally:
            if target_id:
                await manager.close_page(target_id)
            await manager.cleanup()
