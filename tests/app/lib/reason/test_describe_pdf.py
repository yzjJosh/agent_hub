"""Tests for app.lib.reason.describe_pdf module."""

import os
import shutil
from pathlib import Path
from typing import Callable, Self
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from app.lib.reason.describe_pdf import describe_pdf
from tests.app.conftest import is_llm_available
from tests.app.lib.conftest import mock_cli_proc


def _build_mock_cli_side_effect(
    page_count: int,
    text_output: str = "",
    simulate_conversion: bool = False,
    convert_error: bool = False,
) -> Callable[..., MagicMock]:
    """Build a stand in for the tools describe_pdf runs.

    Args:
        page_count: How many pages the PDF is said to have.
        text_output: Text pulled out of the PDF.
        simulate_conversion: Whether page images are really written.
        convert_error: Whether making page images fails.

    Returns:
        A stand in for starting a child process.
    """
    def _mock_run(*cmd: str, **_kw: int) -> MagicMock:
        if cmd[0] == "pdfinfo":
            return mock_cli_proc(stdout=f"Pages:          {page_count}\n")
        if cmd[0] == "pdftotext":
            return mock_cli_proc(stdout=text_output)
        if cmd[0] == "pdftoppm":
            if convert_error:
                return mock_cli_proc(stderr="pdftoppm failed", returncode=1)
            if simulate_conversion:
                output_base = cmd[-1]
                output_file = output_base + f"-{page_count}.jpg"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                Image.new('RGB', (100, 100), color='white').save(output_file)
            return mock_cli_proc()
        return mock_cli_proc()
    return _mock_run


@pytest.fixture
def dummy_pdf(tmp_path: Path) -> str:
    """Create a small valid PDF file."""
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n193\n%%EOF"
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(pdf_content)
    return str(pdf_path)


class TestDescribePdfErrors:
    """Test describe_pdf error handling."""

    async def test_returns_error_for_nonexistent_file(self: Self) -> None:
        """A missing file gives status 2."""
        result, status = await describe_pdf("/nonexistent/path/file.pdf")

        assert status == 2
        assert "does not exist" in result

    async def test_returns_error_for_missing_pdftoppm(self, dummy_pdf: str) -> None:
        """A missing pdftoppm tool gives status 3 and names the tool."""
        with patch('app.lib.reason.describe_pdf.shutil.which') as mock_which:
            mock_which.side_effect = lambda cmd: None if cmd == "pdftoppm" else "/usr/bin/" + cmd
            result, status = await describe_pdf(dummy_pdf)

        assert status == 3
        assert "pdftoppm" in result

    async def test_returns_error_for_missing_pdfinfo(self, dummy_pdf: str) -> None:
        """A missing pdfinfo tool gives status 3 and names the tool."""
        with patch('app.lib.reason.describe_pdf.shutil.which') as mock_which:
            mock_which.side_effect = lambda cmd: None if cmd == "pdfinfo" else "/usr/bin/" + cmd
            result, status = await describe_pdf(dummy_pdf)

        assert status == 3
        assert "pdfinfo" in result

    async def test_returns_error_for_missing_pdftotext(self, dummy_pdf: str) -> None:
        """A missing pdftotext tool gives status 3 and names the tool."""
        with patch('app.lib.reason.describe_pdf.shutil.which') as mock_which:
            mock_which.side_effect = lambda cmd: None if cmd == "pdftotext" else "/usr/bin/" + cmd
            result, status = await describe_pdf(dummy_pdf)

        assert status == 3
        assert "pdftotext" in result

    async def test_returns_error_for_zero_page_count(self, dummy_pdf: str) -> None:
        """A PDF with no pages gives status 1."""
        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec',
                       side_effect=_build_mock_cli_side_effect(page_count=0)):
                result, status = await describe_pdf(dummy_pdf)

        assert status == 1
        assert "page count" in result


class TestDescribePdfTextExtraction:
    """Test pulling text out of a PDF."""

    async def test_returns_extracted_text_when_substantial(self, dummy_pdf: str) -> None:
        """A PDF with plenty of text gives that text back on its own."""
        long_text = "This is a very long text that definitely has more than fifty characters in it."

        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec',
                       side_effect=_build_mock_cli_side_effect(page_count=1, text_output=long_text)):
                result, status = await describe_pdf(dummy_pdf)

        assert status == 0
        assert result == long_text

    @pytest.mark.parametrize("text_output", ["   ", "Short"])
    async def test_falls_back_when_text_insufficient(self, dummy_pdf: str, text_output: str) -> None:
        """A PDF with too little text is described from pictures of its pages."""
        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec',
                       side_effect=_build_mock_cli_side_effect(
                           page_count=1, text_output=text_output, simulate_conversion=True)):
                with patch('app.lib.reason.describe_pdf.describe_image',
                           return_value=("Vision result", 0)) as mock_di:
                    with patch('app.lib.reason.describe_pdf.shutil.rmtree'):
                        result, status = await describe_pdf(dummy_pdf)

        assert status == 0
        assert result == "Vision result"
        mock_di.assert_called_once()


class TestDescribePdfVisionFallback:
    """Test describing a PDF from pictures of its pages."""

    async def test_passes_context_to_describe_image(self, dummy_pdf: str) -> None:
        """The given context is passed along when the pages are described."""
        test_context = "This is test context"

        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec',
                       side_effect=_build_mock_cli_side_effect(
                           page_count=1, text_output="   ", simulate_conversion=True)):
                with patch('app.lib.reason.describe_pdf.describe_image',
                           return_value=("desc", 0)) as mock_di:
                    with patch('app.lib.reason.describe_pdf.shutil.rmtree'):
                        await describe_pdf(dummy_pdf, context=test_context)

        call_args = mock_di.call_args
        assert call_args[0][1] == test_context

    async def test_cleans_up_temp_dir(self, dummy_pdf: str) -> None:
        """The temporary folder is removed when the work is done."""
        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec',
                       side_effect=_build_mock_cli_side_effect(
                           page_count=1, text_output="   ", simulate_conversion=True)):
                with patch('app.lib.reason.describe_pdf.describe_image',
                           return_value=("desc", 0)):
                    with patch('app.lib.reason.describe_pdf.shutil.rmtree') as mock_rmtree:
                        await describe_pdf(dummy_pdf)

        assert mock_rmtree.called

    async def test_returns_error_when_page_conversion_fails(self, dummy_pdf: str) -> None:
        """A page that cannot be turned into a picture gives status 1."""
        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec',
                       side_effect=_build_mock_cli_side_effect(
                           page_count=1, text_output="   ", convert_error=True)):
                with patch('app.lib.reason.describe_pdf.shutil.rmtree'):
                    result, status = await describe_pdf(dummy_pdf)

        assert status == 1
        assert "Could not convert" in result


class TestDescribePdfMultiplePages:
    """Test describe_pdf with multiple page PDFs."""

    async def test_processes_all_pages(self, dummy_pdf: str) -> None:
        """Every page is turned into a picture and all of them are described together."""
        page_count = 3
        converted_pages = []

        def _mock_run(*cmd: str, **_kw: int) -> MagicMock:
            if cmd[0] == "pdfinfo":
                return mock_cli_proc(stdout=f"Pages:          {page_count}\n")
            if cmd[0] == "pdftotext":
                return mock_cli_proc(stdout="   ")
            if cmd[0] == "pdftoppm":
                page_num = int(cmd[2])
                converted_pages.append(page_num)
                output_base = cmd[-1]
                output_file = output_base + f"-{page_num}.jpg"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                Image.new('RGB', (100, 100), color='white').save(output_file)
                return mock_cli_proc()
            return mock_cli_proc()

        with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec', side_effect=_mock_run):
                with patch('app.lib.reason.describe_pdf.describe_image',
                           return_value=("desc", 0)) as mock_di:
                    with patch('app.lib.reason.describe_pdf.shutil.rmtree'):
                        await describe_pdf(dummy_pdf)

        assert converted_pages == [1, 2, 3]
        images_arg = mock_di.call_args[0][0]
        assert len(images_arg) == 3


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@pytest.mark.live
class TestDescribePdfLive:
    """Live tests with a real model server."""

    def _copy_specimen_id(self: Self, tmp_path: Path, data_dir: str) -> str:
        """Copy the made up ID card PDF, which has no text layer, and give back its path."""
        src = os.path.join(data_dir, "specimen_id.pdf")
        dst = tmp_path / "specimen_id.pdf"
        shutil.copy(src, dst)
        return str(dst)

    @pytest.mark.timeout(120)
    async def test_describe_pdf_text(self, tmp_path: Path, data_dir: str) -> None:
        """A PDF with text gets a description."""
        pdf_path = self._copy_specimen_id(tmp_path, data_dir)
        result, status = await describe_pdf(pdf_path)

        assert status == 0
        assert isinstance(result, str) and len(result) > 10

    @pytest.mark.timeout(120)
    async def test_describe_pdf_vision(self, tmp_path: Path, data_dir: str) -> None:
        """A scanned ID card gives back the details printed on it."""
        pdf_path = self._copy_specimen_id(tmp_path, data_dir)
        result, status = await describe_pdf(pdf_path)

        assert status == 0
        upper = result.upper()
        assert "S1234567" in upper
        assert "JANE SAMPLE" in upper
        assert "01/01/1990" in result
        assert "SPECIMEN" in upper
