"""Tests for app.lib.reason.describe_pdf module."""

import os
import shutil
from pathlib import Path
from typing import Callable, Self
from unittest.mock import AsyncMock, patch, MagicMock

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


def _write_text_pdf(path: Path, pages: list[list[str]]) -> str:
    """Write a PDF with a text layer, one list of lines per page.

    Args:
        path: Where the PDF is written.
        pages: The lines of text on each page.

    Returns:
        The path as a string.
    """
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for lines in pages:
        stream = "BT /F1 12 Tf 72 720 Td 16 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    body = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    body += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(body)
    return str(path)


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


_REPORT_TEXT = (
    "Project Falcon report.\nThe budget was signed off by Dana Whitfield.\n\f"
    "Summary.\nThe approved budget for Project Falcon is 48,250 dollars."
)


async def _describe_report(dummy_pdf: str, context: str, reason_reply: tuple[str, int]) -> tuple[str, int, AsyncMock]:
    """Run describe_pdf on a two page PDF with a text layer.

    Args:
        dummy_pdf: Path to the PDF file.
        context: The question to ask.
        reason_reply: What the model gives back.

    Returns:
        The result, the status, and the stand in model call.
    """
    with patch('app.lib.reason.describe_pdf.shutil.which', return_value="/usr/bin/tool"):
        with patch('asyncio.create_subprocess_exec',
                   side_effect=_build_mock_cli_side_effect(page_count=2, text_output=_REPORT_TEXT)):
            with patch('app.lib.reason.describe_image.reason', return_value=reason_reply) as mock_reason:
                result, status = await describe_pdf(dummy_pdf, context)
    return result, status, mock_reason


class TestDescribePdfQuestion:
    """Test answering a question about a PDF with a text layer."""

    @pytest.mark.parametrize("context", ["", "  \n "])
    async def test_empty_question_gives_text(self, dummy_pdf: str, context: str) -> None:
        """With no question, the text is given back and the model is not asked."""
        result, status, mock_reason = await _describe_report(dummy_pdf, context, ("Unused", 0))

        assert (result, status) == (_REPORT_TEXT, 0)
        mock_reason.assert_not_awaited()

    async def test_question_is_answered_from_whole_text(self, dummy_pdf: str) -> None:
        """A question gets one answer made from the text of every page."""
        question = "Who signed off on the budget, and what is the approved amount?"
        result, status, mock_reason = await _describe_report(
            dummy_pdf, f"  {question}  ", (" Dana Whitfield, 48,250 dollars. ", 0),
        )

        assert (result, status) == ("Dana Whitfield, 48,250 dollars.", 0)
        mock_reason.assert_awaited_once()
        prompt, content = mock_reason.call_args.args
        assert prompt.endswith(f"User question: {question}")
        assert "using only the content" in prompt
        assert content == _REPORT_TEXT
        assert "image_files" not in mock_reason.call_args.kwargs

    @pytest.mark.parametrize("reason_reply", [("", 0), ("Backend unavailable", 1)])
    async def test_missing_answer_gives_text(self, dummy_pdf: str, reason_reply: tuple[str, int]) -> None:
        """When no answer comes back, the text is given instead."""
        result, status, mock_reason = await _describe_report(dummy_pdf, "Who signed?", reason_reply)

        assert (result, status) == (_REPORT_TEXT, 0)
        mock_reason.assert_awaited_once()


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

    @pytest.mark.timeout(120)
    async def test_describe_pdf_answers_question_from_scan(self, tmp_path: Path, data_dir: str) -> None:
        """A scanned ID card gets a short answer to a question about it."""
        pdf_path = self._copy_specimen_id(tmp_path, data_dir)
        result, status = await describe_pdf(pdf_path, "What is the date of birth on this card?")

        assert status == 0
        assert "01/01/1990" in result
        assert len(result) < 200

    @pytest.mark.timeout(120)
    async def test_describe_pdf_answers_question_across_pages(self, tmp_path: Path) -> None:
        """A question whose answer is spread over two pages gets one short answer."""
        pdf_path = _write_text_pdf(tmp_path / "report.pdf", [
            ["Project Falcon quarterly report.", "The team finished the design review this quarter."],
            ["Schedule and staffing.", "The budget was signed off by Dana Whitfield, the finance lead."],
            ["Summary.", "The approved budget for Project Falcon is 48,250 dollars."],
        ])
        result, status = await describe_pdf(
            pdf_path, "Who signed off on the budget, and what is the approved amount?"
        )

        assert status == 0
        assert "whitfield" in result.lower()
        assert "48,250" in result
        assert len(result) < 200
