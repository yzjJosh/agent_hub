#!/usr/bin/env python3
"""Test suite for office document conversion in the resolver."""

import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from docx.opc.exceptions import PackageNotFoundError
from PIL import Image
from pptx import Presentation

from app.lib.resolver.office_docs import (
    describe_presentation,
    doc_to_text,
    docx_to_text,
    xls_to_text,
    xlsx_to_text,
)
from tests.app.lib.resolver.conftest import make_docx, make_xls, make_xlsx


def _make_pptx(path: Path) -> Path:
    """Create a real .pptx with two slides."""
    prs = Presentation()
    for title in ("First slide", "Second slide"):
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
    prs.save(str(path))
    return path


def _make_pdf(path: Path) -> Path:
    """Create a real one page PDF."""
    Image.new("RGB", (200, 150), color="white").save(str(path), "PDF")
    return path


_FAKE_SOFFICE_SCRIPT = """#!/bin/bash
# Test double for headless LibreOffice: copies $FAKE_SOURCE_FILE to
# <outdir>/<input stem>.<target>, mimicking `soffice --convert-to <target>`.
outdir=""
target=""
i=0
args=("$@")
while [ $i -lt ${#args[@]} ]; do
  a="${args[$i]}"
  case "$a" in
    --outdir) outdir="${args[$((i+1))]}"; i=$((i+1)) ;;
    --convert-to) target="${args[$((i+1))]}"; i=$((i+1)) ;;
    -*) ;;
    *) input="$a" ;;
  esac
  i=$((i+1))
done
base=$(basename "$input")
stem="${base%.*}"
cp "$FAKE_SOURCE_FILE" "$outdir/$stem.$target"
"""


@pytest.fixture
def fake_soffice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a fake office converter where the code will find it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "soffice"
    script.write_text(_FAKE_SOFFICE_SCRIPT)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return bin_dir


class TestDocxToText:

    async def test_docx_to_text_extracts_paragraphs_and_table(self, tmp_path: Path) -> None:
        """Paragraphs and table cells come out as text with tabs between cells."""
        docx = make_docx(
            tmp_path / "report.docx",
            ["Quarterly report heading", "Body paragraph with details."],
            [["Name", "Value"], ["Revenue", "42"]],
        )

        text = await docx_to_text(str(docx))

        assert "Quarterly report heading" in text
        assert "Body paragraph with details." in text
        assert "Name\tValue" in text
        assert "Revenue\t42" in text

    async def test_docx_to_text_missing_file_raises(self) -> None:
        """A missing file raises PackageNotFoundError."""
        with pytest.raises(PackageNotFoundError):
            await docx_to_text("/nonexistent/file.docx")


class TestXlsxToText:

    async def test_xlsx_to_text_extracts_sheets(self, tmp_path: Path) -> None:
        """Each sheet is named and its cells come out with tabs between them."""
        xlsx = make_xlsx(
            tmp_path / "book.xlsx",
            [("Sales", [["Region", "Total"], ["North", 100]]), ("Notes", [["hello"]])],
        )

        text = await xlsx_to_text(str(xlsx))

        assert "Sheet: Sales" in text
        assert "Region\tTotal" in text
        assert "North\t100" in text
        assert "Sheet: Notes" in text
        assert "hello" in text

    async def test_xlsx_to_text_missing_file_raises(self) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await xlsx_to_text("/nonexistent/file.xlsx")


class TestXlsToText:

    async def test_xls_to_text_extracts_sheets(self, tmp_path: Path) -> None:
        """An old style .xls sheet comes out as text with tabs between cells."""
        xls = make_xls(tmp_path / "legacy.xls", "Data", [["Key", "Value"], ["Count", 7]])

        text = await xls_to_text(str(xls))

        assert "Sheet: Data" in text
        assert "Key\tValue" in text
        assert "Count\t7" in text

    async def test_xls_to_text_missing_file_raises(self) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await xls_to_text("/nonexistent/file.xls")


class TestDocToText:

    @pytest.mark.usefixtures("fake_soffice")
    async def test_doc_to_text_converts_via_libreoffice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An old style .doc comes out as text."""
        source = tmp_path / "source.txt"
        source.write_text("legacy doc text\n")
        monkeypatch.setenv("FAKE_SOURCE_FILE", str(source))
        doc = tmp_path / "legacy.doc"
        doc.write_bytes(b"\xd0\xcf\x11\xe0 dummy legacy doc")

        text = await doc_to_text(str(doc))

        assert text == "legacy doc text"

    @pytest.mark.usefixtures("fake_soffice")
    async def test_doc_to_text_conversion_failure_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A conversion that makes no file raises an error that names soffice."""
        monkeypatch.setenv("FAKE_SOURCE_FILE", str(tmp_path / "missing.txt"))
        doc = tmp_path / "broken.doc"
        doc.write_bytes(b"\xd0\xcf\x11\xe0 broken")

        with pytest.raises(RuntimeError, match="soffice"):
            await doc_to_text(str(doc))


class TestDescribePresentation:

    @pytest.mark.usefixtures("fake_soffice")
    async def test_describe_presentation_uses_vision_flow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Slides are turned into pictures and described."""
        pptx = _make_pptx(tmp_path / "deck.pptx")
        real_pdf = _make_pdf(tmp_path / "deck.pdf")
        monkeypatch.setenv("FAKE_SOURCE_FILE", str(real_pdf))
        mock_describe_image = AsyncMock(return_value=("Slides described.", 0))
        monkeypatch.setattr("app.lib.resolver.office_docs.describe_image", mock_describe_image)

        result, status = await describe_presentation(str(pptx))

        assert status == 0
        assert result == "Slides described."
        images = mock_describe_image.call_args[0][0]
        assert isinstance(images, list)
        assert len(images) == 1

    async def test_describe_presentation_missing_soffice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no office converter, the call gives status 3."""
        pptx = tmp_path / "deck.pptx"
        pptx.write_bytes(b"PK dummy")
        monkeypatch.setattr("app.lib.resolver.office_docs.shutil.which", lambda _name: None)

        result, status = await describe_presentation(str(pptx))

        assert status == 3
        assert "soffice" in result

    async def test_describe_presentation_missing_file(self) -> None:
        """A missing file gives status 2."""
        result, status = await describe_presentation("/nonexistent/deck.pptx")
        assert status == 2
        assert "does not exist" in result

    @pytest.mark.live
    @pytest.mark.timeout(120)
    @pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice not available")
    async def test_describe_presentation_live(self, tmp_path: Path) -> None:
        """A real deck is described with both slide titles in the text."""
        pptx = _make_pptx(tmp_path / "live.pptx")

        result, status = await describe_presentation(str(pptx))

        assert status == 0
        lower = result.lower()
        assert "first slide" in lower
        assert "second slide" in lower
