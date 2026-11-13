"""Describe what is inside a PDF file."""

import asyncio
import os
import shutil
import sys
import uuid

from typing import Optional, Tuple

from app.lib.cli.cli_runner import run_cli
from app.lib.reason.describe_image import answer_question, describe_image
from app.lib.shared import paths


__all__ = ["describe_pdf"]

_SCRATCH_NAME = "pdf_frames"


async def _get_pdf_page_count(pdf_path: str) -> int:
    """Count the pages in a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The number of pages, or 0 when the count cannot be read.
    """
    cmd = ["pdfinfo", pdf_path]
    try:
        result = await run_cli(cmd, "pdfinfo page count")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "pdfinfo failed")
        for line in result.stdout.split('\n'):
            if line.startswith('Pages:'):
                return int(line.split(':')[1].strip())
        return 0
    except Exception as e:
        print(f"Error getting PDF page count: {e}", file=sys.stderr)
        return 0


async def _extract_text_via_pdftotext(pdf_path: str) -> Optional[str]:
    """Pull the text out of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The text, or None when there is none or something goes wrong.
    """
    cmd = ["pdftotext", "-layout", pdf_path, "-"]
    try:
        result = await run_cli(cmd, "pdftotext extraction")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "pdftotext failed")
        text = result.stdout
        return text if text else None
    except Exception as e:
        print(f"Error extracting text via pdftotext: {e}", file=sys.stderr)
        return None


def _rename_if_present(source: str, dest: str) -> bool:
    """Give a file a new name.

    Args:
        source: The file to rename.
        dest: The new name for the file.

    Returns:
        True when the file was there and was renamed.
    """
    if not os.path.exists(source):
        return False
    os.rename(source, dest)
    return True


async def _convert_pdf_page_to_image(pdf_path: str, page_num: int, output_path: str) -> bool:
    """Save one PDF page as an image.

    Args:
        pdf_path: Path to the PDF file.
        page_num: Which page to save, starting at 1.
        output_path: Where the image is written.

    Returns:
        True if the image was written.
    """
    cmd = [
        "pdftoppm",
        "-f", str(page_num),
        "-l", str(page_num),
        "-jpeg",
        "-r", "150",
        pdf_path,
        output_path.replace(".jpg", "")
    ]
    try:
        result = await run_cli(cmd, "pdftoppm page conversion")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "pdftoppm failed")
        expected_path = output_path.replace(".jpg", f"-{page_num}.jpg")
        return await asyncio.to_thread(_rename_if_present, expected_path, output_path)
    except Exception as e:
        print(f"Error converting PDF page to image: {e}", file=sys.stderr)
        return False


async def _check_dependencies() -> Tuple[Optional[str], Optional[int]]:
    """Check that the PDF tools this module needs are installed.

    Returns:
        An error message and status code 3 when a tool is missing, or two
        Nones when they are all there.
    """
    for cmd in ("pdftoppm", "pdfinfo", "pdftotext"):
        if await asyncio.to_thread(shutil.which, cmd) is None:
            return f"{cmd} not found. Please install poppler-utils.", 3
    return None, None


async def _process_vision_fallback(
    pdf_path: str, page_count: int, context: str, verbose: bool
) -> Tuple[str, int]:
    """Describe every page of a PDF from pictures of the pages.

    The page images are deleted before returning.

    Args:
        pdf_path: Path to the PDF file.
        page_count: How many pages the PDF has.
        context: Extra context for the model.
        verbose: If True, prints progress messages.

    Returns:
        The description and a status code. The code is 0 when it worked and
        1 when a page could not be turned into an image.
    """
    run_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(paths.scratch_dir(_SCRATCH_NAME), f"pdf_frames_{run_id}")
    await asyncio.to_thread(os.makedirs, temp_dir, exist_ok=True)
    try:
        images = []
        for i in range(1, page_count + 1):
            frame_path = os.path.join(temp_dir, f"page_{i}.jpg")
            if await _convert_pdf_page_to_image(pdf_path, i, frame_path):
                images.append(frame_path)
            else:
                return f"Could not convert PDF page {i} to image.", 1
        if verbose:
            print(f"Processing PDF via vision: {page_count} pages...")
        return await describe_image(images, context, verbose=verbose)
    finally:
        await asyncio.to_thread(shutil.rmtree, temp_dir)


async def describe_pdf(pdf_path: str, context: str = "", verbose: bool = False) -> Tuple[str, int]:
    """Describe a PDF file, or answer a question about it.

    When the PDF holds enough text and there is no question, that text is
    returned as it is.

    Args:
        pdf_path: Path to the PDF file.
        context: A question about the PDF. Empty asks for its content.
        verbose: If True, prints progress messages.

    Returns:
        The content or the answer, and a status code. The code is 0 when it
        worked, 1 when the PDF could not be read, 2 when the file is
        missing, and 3 when a needed tool is not installed.
    """
    if not await asyncio.to_thread(os.path.exists, pdf_path):
        return f"PDF path '{pdf_path}' does not exist.", 2
    error_msg, error_code = await _check_dependencies()
    if error_msg:
        return error_msg, error_code

    page_count = await _get_pdf_page_count(pdf_path)
    if page_count is None or page_count <= 0:
        return "Could not determine PDF page count.", 1

    if verbose:
        print("Attempting to extract text directly...")

    extracted_text = await _extract_text_via_pdftotext(pdf_path)

    if extracted_text and len(extracted_text.strip()) > 50:
        if verbose:
            print(f"Successfully extracted {len(extracted_text)} characters of text.")
        return await answer_question(extracted_text, context), 0

    if verbose:
        print("Direct text extraction yielded insufficient results. Falling back to vision-based processing.")
    return await _process_vision_fallback(pdf_path, page_count, context, verbose)