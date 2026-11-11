"""Tests for media file probing."""

from pathlib import Path

from app.lib.render.classify import RenderKind
from app.lib.render.probe import probe_meta

FIX = Path(__file__).parent / "fixtures"


async def test_pdf_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes((FIX / "one.pdf").read_bytes())
    meta = await probe_meta(RenderKind.PDF, str(pdf))
    assert meta["pages"] == 1  # the test file has one page


async def test_pdf_missing_is_none(tmp_path: Path) -> None:
    meta = await probe_meta(RenderKind.PDF, str(tmp_path / "nope.pdf"))
    assert meta["pages"] is None


async def test_image_dimensions() -> None:
    png = FIX / "small.png"
    meta = await probe_meta(RenderKind.IMAGE, str(png))
    w, h = meta["dimensions"]
    assert (w, h) == (64, 40)


async def test_image_corrupt_is_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a png")
    meta = await probe_meta(RenderKind.IMAGE, str(p))
    assert meta["dimensions"] is None


async def test_duration_missing_is_none(tmp_path: Path) -> None:
    meta = await probe_meta(RenderKind.VIDEO, str(tmp_path / "nope.mp4"))
    assert meta["duration"] is None


async def test_unknown_kind_empty() -> None:
    assert await probe_meta(RenderKind.MARKDOWN, "/nonexistent") == {}
