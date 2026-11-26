"""Makes thumbnails for files.

Each job runs on its own. A job that fails does not stop later jobs.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass

from PIL import Image

from app.lib.browser.browser_manager import BrowserManager
from app.lib.cli.cli_runner import run_cli
from app.lib.render import cache, gen_state, office
from app.lib.render.classify import RenderKind, classify
from app.lib.shared import config
from app.lib.shared.paths import resolve_path_sync, resources_dir

__all__ = [
    "get_thumbnail_engine",
    "start_directory_thumbnails",
    "start_thumbnail_job",
    "stop_thumbnails",
]

_ENQUEUED: set[str] = set()
_LIVE_JOBS: set[asyncio.Task] = set()
_browser_lock: asyncio.Lock | None = None
_browser_lock_loop: asyncio.AbstractEventLoop | None = None
_render_semaphore: asyncio.Semaphore | None = None
_render_semaphore_loop: asyncio.AbstractEventLoop | None = None

_RENDER_CAP = 4
# Bind hosts that mean every address. The server is reached on loopback then.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_LOOPBACK_HOST = "127.0.0.1"
_CDP_TIMEOUT_MS = 15000
_THUMBNAIL_WIDTH = 1024
_THUMBNAIL_HEIGHT = 640
_DOC_THUMBNAIL_KINDS = frozenset({
    RenderKind.MARKDOWN,
    RenderKind.CODE,
    RenderKind.CSV,
    RenderKind.JSON,
    RenderKind.JSONL,
    RenderKind.YAML,
    RenderKind.XML,
    RenderKind.HTML,
    RenderKind.OFFICE_DOC,
    RenderKind.XLSX,
    RenderKind.XLS,
    RenderKind.TEXT,
})


@dataclass(frozen=True)
class _DirectoryFile:
    """Disk path, resource path, and modified time for one file in a directory."""

    full_path: str
    relative_path: str
    modified_time: int


@dataclass(frozen=True)
class _ThumbnailJob:
    """Cache key, kind, and resource path for a queued thumbnail job."""

    key: str
    kind: RenderKind
    relative_path: str


@dataclass(frozen=True)
class _CropBox:
    """Pixel coordinates for cropping an image to the thumbnail aspect ratio."""

    left: int
    top: int
    right: int
    bottom: int


def get_thumbnail_engine(kind: RenderKind) -> str | None:
    """Return which thumbnail engine handles a kind of file.

    Args:
        kind: What sort of file this is.

    Returns:
        The engine name, or None when this kind gets no thumbnail.
    """
    if kind is RenderKind.IMAGE:
        return "image"
    if kind is RenderKind.VIDEO:
        return "video"
    if kind is RenderKind.PDF:
        return "pdf"
    if kind is RenderKind.OFFICE_PPT:
        return "ppt_pdf"
    if kind in _DOC_THUMBNAIL_KINDS:
        return "doc"
    return None


def start_thumbnail_job(key: str, kind: RenderKind, path: str) -> bool:
    """Start a thumbnail job for one file.

    Nothing happens when the file is already queued or generating, or when
    it gets no thumbnail.

    Args:
        key: The name the file is saved under.
        kind: What sort of file this is.
        path: A path inside the resources folder.

    Returns:
        True when a job was started.
    """
    if key in _ENQUEUED or key in gen_state.generating_keys():
        return False
    if get_thumbnail_engine(kind) is None:
        return False
    _ENQUEUED.add(key)
    gen_state.gen_add(key)
    task = asyncio.get_running_loop().create_task(_run_thumbnail_job(key, kind, path), name=f"thumb-{key[:8]}")
    _LIVE_JOBS.add(task)
    task.add_done_callback(_LIVE_JOBS.discard)
    return True


def _list_directory_files(relative_directory: str) -> list[_DirectoryFile]:
    base_directory = os.path.realpath(resources_dir())
    target = base_directory if not relative_directory else resolve_path_sync(base_directory, relative_directory)
    if target is None or not os.path.isdir(target):
        raise FileNotFoundError(relative_directory)
    found: list[_DirectoryFile] = []
    with os.scandir(target) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            relative_path = os.path.relpath(entry.path, base_directory).replace(os.sep, "/")
            found.append(
                _DirectoryFile(
                    full_path=entry.path,
                    relative_path=relative_path,
                    modified_time=int(entry.stat().st_mtime),
                )
            )
    return found


async def _collect_directory_thumbnail_jobs(relative_directory: str) -> list[_ThumbnailJob]:
    jobs: list[_ThumbnailJob] = []
    for directory_file in await asyncio.to_thread(_list_directory_files, relative_directory):
        kind = await classify(directory_file.full_path)
        if get_thumbnail_engine(kind) is None:
            continue
        key = cache.build_cache_key(directory_file.relative_path, directory_file.modified_time)
        if await cache.get_thumbnail_status(key) in ("pending", "failed"):
            jobs.append(_ThumbnailJob(key=key, kind=kind, relative_path=directory_file.relative_path))
    return jobs


async def start_directory_thumbnails(relative_directory: str) -> int:
    """Start a thumbnail job for every file in one folder that needs one.

    Args:
        relative_directory: A folder inside the resources folder.

    Returns:
        How many jobs were started.
    """
    jobs = await _collect_directory_thumbnail_jobs(relative_directory)
    started = 0
    for job in jobs:
        if start_thumbnail_job(job.key, job.kind, job.relative_path):
            started += 1
    return started


def _resolve_existing_file(path: str) -> str | None:
    full_path = resolve_path_sync(resources_dir(), path)
    if full_path is None or not os.path.isfile(full_path):
        return None
    return full_path


async def _run_thumbnail_job(key: str, kind: RenderKind, path: str) -> None:
    try:
        full_path = await asyncio.to_thread(_resolve_existing_file, path)
        if full_path is None:
            raise FileNotFoundError(path)
        engine = get_thumbnail_engine(kind)
        if engine in ("doc", "ppt_pdf"):
            if await office.ensure_doc_for_thumb(kind, key, path, full_path) != "published":
                raise RuntimeError("document unavailable")
        async with _get_render_semaphore():
            if engine == "image":
                await _render_image_thumbnail(full_path, key)
            elif engine == "video":
                await _render_video_thumbnail(full_path, key)
            elif engine == "pdf":
                await _render_pdf_thumbnail(full_path, key)
            elif engine == "ppt_pdf":
                pdf_path = os.path.join(cache.get_render_root(), key, cache.OFFICE_PDF_ASSET)
                await _render_pdf_thumbnail(pdf_path, key)
            elif engine == "doc":
                await _render_document_thumbnail(key)
            else:
                raise RuntimeError(f"no engine for {kind}")
    except Exception as error:
        await cache.mark_failed(key, "thumb", (str(error) or "thumbnail failed")[:200])
    finally:
        gen_state.gen_remove(key)
        _ENQUEUED.discard(key)


async def stop_thumbnails() -> None:
    """Stop thumbnail making at shutdown."""
    _ENQUEUED.clear()
    _LIVE_JOBS.clear()


def _get_render_semaphore() -> asyncio.Semaphore:
    global _render_semaphore, _render_semaphore_loop
    loop = asyncio.get_running_loop()
    if _render_semaphore is None or _render_semaphore_loop is not loop:
        _render_semaphore = asyncio.Semaphore(_RENDER_CAP)
        _render_semaphore_loop = loop
    return _render_semaphore


def _get_browser_lock() -> asyncio.Lock:
    global _browser_lock, _browser_lock_loop
    loop = asyncio.get_running_loop()
    if _browser_lock is None or _browser_lock_loop is not loop:
        _browser_lock = asyncio.Lock()
        _browser_lock_loop = loop
    return _browser_lock


def _get_view_base_url() -> str:
    host, port = config.http_bind()
    if host in _WILDCARD_HOSTS:
        host = _LOOPBACK_HOST
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _compute_cover_crop(
    width: int, height: int, ratio_width: int = 16, ratio_height: int = 10
) -> _CropBox:
    target_ratio = ratio_width / ratio_height
    if width / height > target_ratio:
        cropped_width = int(height * target_ratio)
        left = (width - cropped_width) // 2
        return _CropBox(left=left, top=0, right=left + cropped_width, bottom=height)
    cropped_height = int(width / target_ratio)
    top = (height - cropped_height) // 2
    return _CropBox(left=0, top=top, right=width, bottom=top + cropped_height)


def _save_png_atomically(image: Image.Image, key: str) -> None:
    key_directory = os.path.join(cache.get_render_root(), key)
    os.makedirs(key_directory, exist_ok=True)
    staging_path = os.path.join(key_directory, f".thumb-{os.getpid()}-{threading.get_ident()}.png")
    image.save(staging_path, "PNG")
    os.replace(staging_path, os.path.join(key_directory, "thumb.png"))


def _crop_and_resize_image(image: Image.Image) -> Image.Image:
    crop_box = _compute_cover_crop(image.width, image.height)
    crop_bounds = (crop_box.left, crop_box.top, crop_box.right, crop_box.bottom)
    return image.crop(crop_bounds).resize((_THUMBNAIL_WIDTH, _THUMBNAIL_HEIGHT))


def _save_cropped_thumbnail_sync(
    source_path: str, key: str, *, should_remove_source: bool = False
) -> None:
    with Image.open(source_path) as image:
        image.load()
        _save_png_atomically(_crop_and_resize_image(image), key)
    if should_remove_source:
        os.remove(source_path)


async def _render_image_thumbnail(full_path: str, key: str) -> None:
    await asyncio.to_thread(_save_cropped_thumbnail_sync, full_path, key)


async def _render_video_thumbnail(full_path: str, key: str) -> None:
    probe_result = await run_cli(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            full_path,
        ],
        "ffprobe",
        timeout=10,
    )
    seek_seconds = 0.0
    if probe_result.returncode == 0 and probe_result.stdout:
        try:
            seek_seconds = min(float(probe_result.stdout) * 0.1, 5.0)
        except ValueError:
            pass
    key_directory = os.path.join(cache.get_render_root(), key)
    await asyncio.to_thread(os.makedirs, key_directory, exist_ok=True)
    frame_path = os.path.join(key_directory, ".frame.png")
    result = await run_cli(
        ["ffmpeg", "-y", "-ss", f"{seek_seconds}", "-i", full_path, "-frames:v", "1", frame_path],
        "ffmpeg",
        timeout=30,
    )
    if result.returncode != 0 or not await asyncio.to_thread(os.path.exists, frame_path):
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")
    await asyncio.to_thread(_save_cropped_thumbnail_sync, frame_path, key, should_remove_source=True)


def _list_pdf_page_files(key_directory: str) -> list[str]:
    return sorted(filename for filename in os.listdir(key_directory) if filename.startswith(".page-"))


async def _render_pdf_thumbnail(full_path: str, key: str) -> None:
    key_directory = os.path.join(cache.get_render_root(), key)
    await asyncio.to_thread(os.makedirs, key_directory, exist_ok=True)
    prefix = os.path.join(key_directory, ".page")
    result = await run_cli(
        ["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1", full_path, prefix],
        "pdftoppm",
        timeout=30,
    )
    pages = await asyncio.to_thread(_list_pdf_page_files, key_directory)
    if result.returncode != 0 or not pages:
        raise RuntimeError(f"pdftoppm failed: {result.stderr[-200:]}")
    page_path = os.path.join(key_directory, pages[0])
    await asyncio.to_thread(_save_cropped_thumbnail_sync, page_path, key, should_remove_source=True)


async def _render_document_thumbnail(key: str) -> None:
    async with _get_browser_lock():
        manager = await BrowserManager.create()
        target_id: str | None = None
        try:
            target_id = await manager.open_page()
            is_viewport_set = await manager.set_viewport_size(target_id, _THUMBNAIL_WIDTH, _THUMBNAIL_HEIGHT)
            if not is_viewport_set:
                raise RuntimeError("thumbnail viewport failed")
            page_url = f"{_get_view_base_url()}/view/doc/{key}?thumb=1"
            is_loaded = await manager.navigate(target_id, page_url, wait_until="load", timeout=_CDP_TIMEOUT_MS)
            if not is_loaded:
                raise RuntimeError("thumbnail page did not load")
            key_directory = os.path.join(cache.get_render_root(), key)
            staging_path = os.path.join(key_directory, f".cdp-{os.getpid()}.jpg")
            await asyncio.to_thread(os.makedirs, key_directory, exist_ok=True)
            saved_path = await manager.capture_screenshot(target_id, staging_path, full_page=False)
            if saved_path is None:
                raise RuntimeError("thumbnail screenshot failed")
            await asyncio.to_thread(_save_cropped_thumbnail_sync, staging_path, key, should_remove_source=True)
        finally:
            if target_id:
                await manager.close_page(target_id)
            await manager.cleanup()
