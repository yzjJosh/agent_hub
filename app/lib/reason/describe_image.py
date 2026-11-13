#!/usr/bin/env python3
"""Describe what is in an image."""

__all__ = ["answer_question", "describe_image"]

import asyncio
import base64
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import httpx
import pillow_heif
import reverse_geocoder
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

from app.lib.cli.cli_runner import run_cli
from app.lib.shared.model_paths import get_models_dir
from app.lib.reason import rotation_detector
from app.lib.reason.document_classifier import is_document
from app.lib.reason.reason import reason
from app.lib.shared import config, paths


def _ensure_ocr_error_model() -> str:
    """Make sure the OCR check model is on disk.

    Returns:
        The folder holding the model.
    """
    model_dir = Path(get_models_dir()) / "OCRerrcr"
    if not model_dir.exists():
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download("PleIAs/OCRerrcr", local_dir=str(model_dir))
    return str(model_dir)


# The OCR check model is loaded once and kept here.
_OCR_ERROR_MODEL = None
_OCR_ERROR_TOKENIZER = None
# OCR text is thrown away when more of it than this looks wrong.
_OCR_ERROR_THRESHOLD = 0.15

# HEIC and HEIF images can be opened after this call.
pillow_heif.register_heif_opener()

_DEFAULT_BATCH_SIZE = 3
_SCRATCH_NAME = "images"
_OCR_TIMEOUT_S = 600

_ANSWER_PROMPT = """Answer the user question below using only the content above. The content is a transcription or description of a file.
CRITICAL INSTRUCTIONS:
1. ANSWER THE QUESTION: Answer exactly what the user asks. Do not describe or summarize the whole content unless the question asks for it.
2. NOT SHOWN: If the content does not show or contain what the question asks about, say so in the answer instead of guessing. Never fill gaps with general knowledge.
3. EXACT TEXT: Copy text, names, and numbers verbatim from the content when the question asks for them or when they are the answer.
4. METADATA: A line starting with "[Metadata:" gives the camera, date, and place of a photo. Use it when the question asks about these. When the question asks for a general description, start the answer with that line as it is.
5. OUTPUT FORMAT: Output only the answer, without conversational filler, greetings, or closing remarks.
"""


def _has_sufficient_content(text: str, min_chars: int = 100) -> bool:
    """Check that OCR text has enough real content.

    Only letters, Chinese characters, and numbers count.

    Args:
        text: The OCR text to check.
        min_chars: How many characters must count.

    Returns:
        True if there are enough.
    """
    if not text:
        return False
    # Text inside comment markers does not count.
    cleaned = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    valid_chars = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]", cleaned)

    return len(valid_chars) >= min_chars


def _parse_exposure_time(val: int | float | str) -> Optional[float]:
    """Read an exposure time that may be written as a fraction.

    Args:
        val: The exposure time as a number or as text.

    Returns:
        The exposure time in seconds, or None when it cannot be read.
    """
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        if isinstance(val, str) and "/" in val:
            try:
                parts = val.split("/")
                return float(parts[0]) / float(parts[1])
            except (ValueError, IndexError, ZeroDivisionError):
                return None
        return None


def _validate_ocr_quality(text: str, verbose: bool = False) -> bool:
    """Check whether OCR text is clean enough to keep.

    The text is kept when the check itself cannot run.

    Args:
        text: The OCR text to check.
        verbose: If True, prints extra messages.

    Returns:
        True if little enough of the text looks wrong.
    """
    global _OCR_ERROR_MODEL, _OCR_ERROR_TOKENIZER

    if not text or not text.strip():
        return True

    try:
        if _OCR_ERROR_MODEL is None or _OCR_ERROR_TOKENIZER is None:
            try:
                model_name = _ensure_ocr_error_model()
                if verbose:
                    print(f"[DEBUG] Loading OCR error model from: {model_name}")

                _OCR_ERROR_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
                _OCR_ERROR_MODEL = AutoModelForTokenClassification.from_pretrained(
                    model_name, device_map="cpu"
                )
                _OCR_ERROR_MODEL.eval()
            except Exception as e:
                print(f"[ERROR] Failed to load OCR error model: {e}", file=sys.stderr)
                return True

        error_ratio = _compute_ocr_error_ratio(text)
        if error_ratio is None:
            return True
        if verbose and error_ratio > 0:
            print(f"[DEBUG] OCR error ratio: {error_ratio:.2%}")
        if error_ratio > _OCR_ERROR_THRESHOLD:
            print(
                f"[WARN] OCR quality validation failed: {error_ratio:.2%} error rate "
                f"exceeds {_OCR_ERROR_THRESHOLD:.0%} threshold",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:
        print(f"[ERROR] OCR quality validation error: {e}", file=sys.stderr)
        return True


def _compute_ocr_error_ratio(text: str) -> Optional[float]:
    """Work out how much of the text looks wrong.

    Args:
        text: The text to check.

    Returns:
        The share of the text that looks wrong, from 0 to 1, or None when
        the check cannot run.
    """
    try:
        inputs = _OCR_ERROR_TOKENIZER(  # type: ignore[possibly-undefined]
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = _OCR_ERROR_MODEL(**inputs)  # type: ignore[possibly-undefined]
            predictions = torch.argmax(outputs.logits, dim=-1)

        error_count = (predictions == 1).sum().item()
        total_tokens = predictions.numel()
        if total_tokens == 0:
            return None
        return error_count / total_tokens
    except Exception as e:
        print(f"[ERROR] OCR inference error: {e}", file=sys.stderr)
        return None


async def _extract_exif_metadata(image_path: str) -> Dict:
    """Read the extra info stored inside an image.

    Args:
        image_path: Path to the original image file.

    Returns:
        The fields found, such as camera, date, place, exposure, and size.
        Only the size is there when the rest cannot be read, and nothing is
        there when the image cannot be read at all.
    """
    try:
        result = await run_cli(
            ["exiftool", "-json", "-n", image_path], "exiftool exif extraction", timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError(f"exiftool failed with return code {result.returncode}")

        return await asyncio.to_thread(_parse_exif_json, result.stdout)
    except Exception:
        return await asyncio.to_thread(_extract_exif_fallback, image_path)


def _extract_exif_fallback(image_path: str) -> Dict:
    """Read just the width and height of an image.

    This blocks, so callers must run it off the event loop.

    Args:
        image_path: Path to the image file.

    Returns:
        The width and height, or nothing when the image cannot be read.
    """
    try:
        with Image.open(image_path) as img:
            return {"image_width": img.width, "image_height": img.height}
    except Exception:
        return {}


def _parse_exif_json(stdout: str) -> Dict:
    """Pick the wanted fields out of an image info dump.

    This blocks, so callers must run it off the event loop.

    Args:
        stdout: The raw info dump as JSON text.

    Returns:
        The fields found, or nothing when there are none.
    """
    data = json.loads(stdout)
    if not data or not isinstance(data, list) or len(data) == 0:
        return {}

    exif = data[0]
    result_dict = {}
    result_dict.update(_parse_basic_exif(exif))
    result_dict.update(_parse_gps_location(exif))
    return {k: v for k, v in result_dict.items() if v is not None}


def _parse_basic_exif(exif: Dict) -> Dict:
    """Pick the plain fields out of image info.

    Args:
        exif: The image info to read.

    Returns:
        The fields found, such as size, camera, date, and exposure.
    """
    result = {}
    if "ImageWidth" in exif:
        result["image_width"] = exif["ImageWidth"]
    if "ImageHeight" in exif:
        result["image_height"] = exif["ImageHeight"]
    if "Make" in exif:
        result["camera_make"] = exif["Make"].strip() if exif["Make"] else None
    if "Model" in exif:
        result["camera_model"] = exif["Model"].strip() if exif["Model"] else None
    if "DateTimeOriginal" in exif:
        result["datetime_original"] = (
            exif["DateTimeOriginal"].strip() if exif["DateTimeOriginal"] else None
        )
    if "DateTime" in exif:
        result["datetime"] = exif["DateTime"].strip() if exif["DateTime"] else None
    if "ExposureTime" in exif:
        result["exposure_time"] = _parse_exposure_time(exif["ExposureTime"])
    if "FNumber" in exif:
        result["f_number"] = (
            float(exif["FNumber"]) if exif["FNumber"] is not None else None
        )
    if "ISO" in exif:
        result["iso"] = int(exif["ISO"]) if exif["ISO"] is not None else None
    if "FocalLength" in exif:
        result["focal_length"] = (
            float(exif["FocalLength"]) if exif["FocalLength"] is not None else None
        )
    if "Orientation" in exif:
        result["orientation"] = (
            int(exif["Orientation"]) if exif["Orientation"] is not None else None
        )
    return result


def _parse_gps_location(exif: Dict) -> Dict:
    """Pick the place out of image info.

    Args:
        exif: The image info to read.

    Returns:
        The latitude and longitude when they are there, plus the city,
        state, county, and country they fall in.
    """
    result = {}
    lat = exif.get("GPSLatitude")
    lon = exif.get("GPSLongitude")
    if lat is not None:
        result["latitude"] = float(lat)
    if lon is not None:
        result["longitude"] = float(lon)
    if lat is not None and lon is not None:
        try:
            geo_result = reverse_geocoder.search((lat, lon), verbose=False, mode=1)
            if geo_result and len(geo_result) > 0:
                location = geo_result[0]
                result["city"] = location.get("name")
                result["state"] = location.get("admin1")
                result["county"] = location.get("admin2")
                result["country_code"] = location.get("cc")
        except Exception:
            pass
    return result


@dataclass
class _ImageInfo:
    """One image and what has been found out about it."""

    index: int
    original_path: str
    temp_path: str
    exif_metadata: Dict = None

    def __post_init__(self) -> None:
        if self.exif_metadata is None:
            self.exif_metadata = {}


def _convert_image(input_path: str, output_path: str) -> bool:
    """Save an image as a JPG.

    Args:
        input_path: Path to the input image file.
        output_path: Path where the JPG is written.

    Returns:
        True if the JPG was written.
    """
    try:
        with Image.open(input_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(output_path, "JPEG", quality=85)
            return True

    except Exception as e:
        print(f"[ERROR] Error converting image {input_path}: {e}", file=sys.stderr)
        return False


async def _prepare_image(
    index: int, source_path: str, temp_path: str, verbose: bool
) -> Optional[_ImageInfo]:
    """Save an image as an upright JPG.

    Args:
        index: Position of the image in the caller's list.
        source_path: Path to the original image file.
        temp_path: Path where the JPG is written.
        verbose: If True, prints extra messages.

    Returns:
        What is known about the image, or None when the JPG could not be
        written.
    """
    if not await asyncio.to_thread(_convert_image, source_path, temp_path):
        print(f"[ERROR] Failed to convert image {source_path}", file=sys.stderr)
        return None

    await rotation_detector.rotate_images_in_place([temp_path], verbose=verbose)

    return _ImageInfo(
        index=index,
        original_path=source_path,
        temp_path=temp_path,
    )


def _resolve_ocr_backend() -> Optional[config.Backend]:
    """Find the OCR server.

    Returns:
        The OCR server, or None when it is not set.
    """
    try:
        return config.backend(config.OCR)
    except config.MissingSettingError:
        return None


async def _process_ocr(
    temp_path: str, index: int, ocr: config.Backend, verbose: bool = False
) -> Tuple[int, str, bool]:
    """Read the text off one image.

    Args:
        temp_path: Path to the image file.
        index: Position of the image in the caller's list.
        ocr: The OCR server to ask.
        verbose: If True, prints extra messages.

    Returns:
        The position, the text, and whether it worked. The text is empty
        when it did not.
    """
    start_time = time.time()
    if verbose:
        print(f"[DEBUG] Starting OCR for image {index}: {temp_path}")

    try:
        image_data = await asyncio.to_thread(
            _encode_image_file, temp_path, index, verbose
        )
        response = await _send_ocr_request(ocr, image_data)

        if response.status_code != 200:
            _log_ocr_error(index, response)
            return (index, "", False)

        text = await asyncio.to_thread(
            _extract_and_validate_ocr_text, response, index, verbose
        )
        if text is None:
            return (index, "", False)

        if verbose:
            print(f"[DEBUG] Image {index}: Total time {time.time() - start_time:.2f}s")
        return (index, text, True)
    except Exception as e:
        print(
            f"[ERROR] Image {index}: GLM-OCR error after "
            f"{time.time() - start_time:.2f}s: {e}",
            file=sys.stderr,
        )
        return (index, "", False)


def _encode_image_file(temp_path: str, index: int, verbose: bool) -> str:
    """Read an image as text that can be sent to the OCR server.

    Args:
        temp_path: Path to the image file.
        index: Position of the image in the caller's list.
        verbose: If True, prints extra messages.

    Returns:
        The image as text.
    """
    encode_start = time.time()
    with open(temp_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    if verbose:
        print(
            f"[DEBUG] Image {index}: Encoded {len(image_data)} bytes "
            f"in {time.time() - encode_start:.2f}s"
        )
    return image_data


def _log_ocr_error(index: int, response: httpx.Response) -> None:
    """Print why a request to the OCR server failed.

    Args:
        index: Position of the image in the caller's list.
        response: The reply from the OCR server.
    """
    print(
        f"[ERROR] Image {index}: OCR failed with status {response.status_code}",
        file=sys.stderr,
    )
    print(
        f"[ERROR] Image {index}: Response body: {response.text[:500]}",
        file=sys.stderr,
    )


def _extract_and_validate_ocr_text(
    response: httpx.Response, index: int, verbose: bool
) -> Optional[str]:
    """Get the text out of an OCR reply and check it is good enough.

    Args:
        response: The reply from the OCR server.
        index: Position of the image in the caller's list.
        verbose: If True, prints extra messages.

    Returns:
        The text, or None when it is missing, too short, or too messy.
    """
    text = _extract_ocr_text(response)
    if text is None:
        return None

    if verbose:
        print(f"[DEBUG] Image {index}: OCR completed, extracted {len(text)} chars")

    if not _has_sufficient_content(text):
        print(
            f"[WARN] Image {index}: OCR output too short ({len(text)} chars), "
            f"falling back to LLM vision",
            file=sys.stderr,
        )
        return None

    validate_start = time.time()
    if not _validate_ocr_quality(text, verbose=verbose):
        print(
            f"[ERROR] Image {index}: OCR quality validation failed in "
            f"{time.time() - validate_start:.2f}s",
            file=sys.stderr,
        )
        return None

    if verbose:
        print(
            f"[DEBUG] Image {index}: Quality validation passed in "
            f"{time.time() - validate_start:.2f}s"
        )
    return text


async def _send_ocr_request(ocr: config.Backend, image_data: str) -> httpx.Response:
    """Ask the OCR server to read one image.

    The model is named and the key is sent only when they are set.

    Args:
        ocr: The OCR server to ask.
        image_data: The image as text.

    Returns:
        The reply from the OCR server.

    Raises:
        httpx.HTTPError: If the server cannot be reached.
    """
    async with httpx.AsyncClient(
        timeout=_OCR_TIMEOUT_S, follow_redirects=True
    ) as client:
        return await client.post(
            f"{ocr.url}/chat/completions",
            json=_build_ocr_payload(ocr, image_data),
            headers=ocr.auth_headers(),
        )


def _build_ocr_payload(ocr: config.Backend, image_data: str) -> dict:
    """Build the request body that asks the OCR server to read one image.

    Args:
        ocr: The OCR server to ask.
        image_data: The image as text.

    Returns:
        The request body.
    """
    payload: dict = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the provided document into Markdown. "
                            "You must capture the peripheral text (headers, footers, "
                            "page numbers) by placing them inside blockquotes at the "
                            "start and end of the transcription.\n"
                            "- Scan the top margin first. If there is text, start your "
                            "response with: `> Top Margin: [text]`\n"
                            "- Transcribe the main text using standard Markdown headers "
                            "(`#`, `##`) for visual titles.\n"
                            "- Scan the bottom margin last. If there is text, end your "
                            "response with: `> Bottom Margin: [text]`"
                        ),
                    },
                ],
            }
        ]
    }
    if ocr.model:
        payload["model"] = ocr.model
    return payload


def _extract_ocr_text(response: httpx.Response) -> Optional[str]:
    """Get the text out of an OCR reply.

    Args:
        response: The reply from the OCR server.

    Returns:
        The text, or None when the reply has none.
    """
    data = response.json()
    if not data.get("choices"):
        print("[ERROR] No choices in OCR response", file=sys.stderr)
        return None
    text = data["choices"][0]["message"]["content"]
    if not text or not text.strip():
        print("[ERROR] Empty OCR output", file=sys.stderr)
        return None
    return text


class _ImagePipeline:
    """Turns images into descriptions."""

    def __init__(self, batch_size: int = _DEFAULT_BATCH_SIZE, verbose: bool = False) -> None:
        self.batch_size = batch_size
        self.verbose = verbose

    async def process(
        self, image_paths: Union[str, List[str]], context: str = ""
    ) -> Tuple[List[str], int]:
        """Describe images.

        The working files are deleted before returning.

        Args:
            image_paths: One path or a list of paths to image files.
            context: Extra context for the model.

        Returns:
            One description per image and a status code. The code is 0 when
            it worked, 1 when the model failed, and 2 when an image could
            not be read.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not image_paths:
            return [], 0

        image_count = len(image_paths)
        run_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join(paths.scratch_dir(_SCRATCH_NAME), f"images_{run_id}")
        await asyncio.to_thread(os.makedirs, temp_dir, exist_ok=True)

        try:
            return await self._run_pipeline(image_paths, image_count, temp_dir, context)
        finally:
            await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    async def _run_pipeline(
        self,
        image_paths: List[str],
        image_count: int,
        temp_dir: str,
        context: str,
    ) -> Tuple[List[str], int]:
        """Describe every image and put the results back in order.

        Args:
            image_paths: Paths to the image files.
            image_count: How many images there are.
            temp_dir: Folder the working files go in.
            context: Extra context for the model.

        Returns:
            One description per image and a status code. The code is 0 when
            it worked, 1 when the model failed, and 2 when an image could
            not be read.
        """
        infos = await self._preprocess(image_paths, temp_dir)
        if len(infos) != image_count:
            return [], 2

        documents, pictures = await self._classify(infos)
        exif_by_index = {info.index: info.exif_metadata for info in infos}
        results, error_status = await self._process_images(
            documents, pictures, context, exif_by_index
        )
        if error_status == 1:
            return [], 1

        descriptions, reassemble_status = self._reassemble(
            results, image_count, exif_by_index
        )
        if reassemble_status == 1:
            return [], 1
        return descriptions, 0

    async def _preprocess(
        self, image_paths: List[str], temp_dir: str
    ) -> List[_ImageInfo]:
        """Save each image as an upright JPG and read its extra info.

        Args:
            image_paths: Paths to the original images.
            temp_dir: Folder the JPGs go in.

        Returns:
            What is known about each image that could be saved.
        """
        infos = []

        for i, path in enumerate(image_paths):
            temp_path = os.path.join(temp_dir, f"image_{i}.jpg")
            info = await _prepare_image(i, path, temp_path, self.verbose)
            if info is not None:
                info.exif_metadata = await _extract_exif_metadata(path)
                infos.append(info)

        return infos

    async def _classify(
        self, infos: List[_ImageInfo]
    ) -> Tuple[List[_ImageInfo], List[_ImageInfo]]:
        """Split images into documents and pictures.

        An image that cannot be sorted counts as a picture.

        Args:
            infos: The images to sort.

        Returns:
            The documents and the pictures.
        """
        documents = []
        pictures = []

        for info in infos:
            try:
                if await is_document(info.temp_path, verbose=self.verbose):
                    documents.append(info)
                else:
                    pictures.append(info)
            except Exception:
                print(
                    f"[WARN] Classification error for {info.temp_path}, defaulting to picture",
                    file=sys.stderr,
                )
                pictures.append(info)

        return documents, pictures

    async def _process_images(
        self,
        documents: List[_ImageInfo],
        pictures: List[_ImageInfo],
        context: str,
        exif_by_index: Dict[int, Dict],
    ) -> Tuple[Dict[int, str], int]:
        """Describe the documents and the pictures.

        A document whose text cannot be read is described like a picture.

        Args:
            documents: The document images.
            pictures: The picture images.
            context: Extra context for the model.
            exif_by_index: The extra info for each image, by position.

        Returns:
            The description for each position and a status code. The code is
            0 when it worked and 1 when the model failed.
        """
        ocr_results = {}
        to_llm_indices = []

        index_to_path = {info.index: info.temp_path for info in documents + pictures}

        ocr = _resolve_ocr_backend() if documents else None
        for doc in documents:
            if ocr is None:
                to_llm_indices.append(doc.index)
                continue
            idx, text, success = await _process_ocr(
                doc.temp_path, doc.index, ocr, self.verbose
            )
            if success:
                ocr_results[idx] = text
            else:
                to_llm_indices.append(idx)

        for pic in pictures:
            to_llm_indices.append(pic.index)

        to_llm_indices.sort()

        llm_results = await self._llm_consumer(
            to_llm_indices, context, index_to_path, exif_by_index
        )

        if llm_results is None and to_llm_indices:
            return {}, 1

        return {**ocr_results, **(llm_results or {})}, 0

    async def _llm_consumer(
        self,
        pending_indices: List[int],
        context: str,
        index_to_path: Dict[int, str],
        exif_by_index: Dict[int, Dict],
    ) -> Optional[Dict[int, str]]:
        """Describe images with the model, a few at a time.

        Args:
            pending_indices: Positions of the images to describe.
            context: Extra context for the model.
            index_to_path: The image file for each position.
            exif_by_index: The extra info for each image, by position.

        Returns:
            The description for each position, or None when the model
            failed.
        """
        if not pending_indices:
            return {}
        local_results = {}

        for i in range(0, len(pending_indices), self.batch_size):
            batch_indices = pending_indices[i : i + self.batch_size]
            batch_paths = [index_to_path[idx] for idx in batch_indices]
            batch_context = self._build_batch_context(
                context, batch_indices, exif_by_index
            )
            batch_prompt = self._build_llm_prompt(len(batch_paths))

            desc, desc_status = await reason(
                batch_prompt, batch_context, thinking=False, image_files=batch_paths
            )
            if desc_status != 0:
                return None

            parsed = self._parse_batch_results(desc, batch_indices)
            if parsed is None:
                return None
            local_results.update(parsed)

            if self.verbose:
                print(
                    f"Processed {min(i + self.batch_size, len(pending_indices))}/"
                    f"{len(pending_indices)} LLM images..."
                )

        return local_results

    def _build_batch_context(
        self,
        context: str,
        batch_indices: List[int],
        exif_by_index: Dict[int, Dict],
    ) -> str:
        """Build the context text for a few images.

        Args:
            context: Extra context for the model.
            batch_indices: Positions of the images in this batch.
            exif_by_index: The extra info for each image, by position.

        Returns:
            The context text with the extra info added.
        """
        metadata_lines = []
        for j, idx in enumerate(batch_indices):
            exif = exif_by_index.get(idx, {})
            metadata_str = self._build_metadata_prefix(exif)
            if metadata_str:
                metadata_lines.append(f"Image {j+1}: {metadata_str}")
        if metadata_lines:
            return f"{context}\n\nMetadata:\n" + "\n".join(metadata_lines)
        return context

    def _build_llm_prompt(self, num_images: int) -> str:
        """Build the question asked about a few images.

        Args:
            num_images: How many images are asked about.

        Returns:
            The question text.
        """
        return (
            f"Analyze these {num_images} images. "
            "Provide a detailed, structured description.\n"
            "For each image:\n"
            "    - If it's a screenshot or a document:\n"
            "        - Include a word-to-word transcription of all its text content "
            "in markdown format. Follow the original layout, including headings, "
            "paragraphs, lists, tables, statements. Every details matter. "
            "Don't miss anything.\n"
            "        - Only include the raw content from the image. Don't include "
            "anything not in the image (file name, image number, metadata).\n"
            "    - Otherwise, include:\n"
            "        - Detailed description of the picture in details. Include "
            "everything you see. Don't miss any details.\n"
            "        - If the picture has text, transcribe it following the "
            "document transcription instruction.\n"
            "Be literal and thorough. Don't interpret or summarize. "
            "Just describe what you see.\n"
            "Process images one by one.\n"
            "CRITICAL OUTPUT FORMAT REQUIREMENTS:\n"
            '    1. If there are multiple images, descriptions should be separated '
            'by "==================".\n'
            "    2. DO NOT put image number or titles.\n"
            "    2. No conversational filler, no greetings, no 'here is the "
            "description,' and no closing remarks. Output only the data requested."
        )

    def _parse_batch_results(
        self, desc: str, batch_indices: List[int]
    ) -> Optional[Dict[int, str]]:
        """Split one answer into a description per image.

        Args:
            desc: The answer from the model.
            batch_indices: Positions of the images in this batch.

        Returns:
            The description for each position, or None when one is missing
            or empty.
        """
        batch_results = re.split("=+\n", desc)
        local_results = {}
        for j, idx in enumerate(batch_indices):
            if j >= len(batch_results):
                return None
            result = batch_results[j]
            if not result or not result.strip():
                return None
            local_results[idx] = result.strip()
        return local_results

    def _build_metadata_prefix(self, exif: Dict) -> str:
        """Build a short line about where and how an image was taken.

        Args:
            exif: The extra info for one image.

        Returns:
            The line, or an empty string when there is nothing to say.
        """
        if not exif:
            return ""

        parts = []
        make = exif.get("camera_make", "")
        model = exif.get("camera_model", "")
        if make or model:
            parts.append(f"Camera: {make} {model}".strip())
        date = exif.get("datetime_original") or exif.get("datetime")
        if date:
            parts.append(f"Date: {date}")
        location = self._format_location(exif)
        if location:
            parts.append(f"Location: {location}")
        parts.extend(self._format_camera_settings(exif))
        return ", ".join(parts)

    def _format_location(self, exif: Dict) -> str:
        """Build the place part of the line about an image.

        Args:
            exif: The extra info for one image.

        Returns:
            The place name, the latitude and longitude when there is no
            name, or an empty string.
        """
        city = exif.get("city")
        state = exif.get("state")
        county = exif.get("county")
        country_code = exif.get("country_code")
        if city and state:
            location_str = f"{city}, {state}"
            if county and county != state:
                location_str += f" ({county})"
            if country_code:
                location_str += f", {country_code}"
            return location_str
        if city:
            return city
        if "latitude" in exif and "longitude" in exif:
            return f"{exif['latitude']:.4f}, {exif['longitude']:.4f}"
        return ""

    def _format_camera_settings(self, exif: Dict) -> List[str]:
        """Build the camera setting parts of the line about an image.

        Args:
            exif: The extra info for one image.

        Returns:
            One piece of text for each setting that is there.
        """
        parts = []
        if "iso" in exif:
            parts.append(f"ISO: {exif['iso']}")
        if "f_number" in exif:
            parts.append(f"f/{exif['f_number']}")
        if "exposure_time" in exif:
            exp = exif["exposure_time"]
            if 0 < exp < 1:
                parts.append(f"Exposure: 1/{int(1/exp)}")
            else:
                parts.append(f"Exposure: {exp}")
        return parts

    def _reassemble(
        self,
        results: Dict[int, str],
        total: int,
        exif_by_index: Dict[int, Dict] = None,
    ) -> Tuple[List[str], int]:
        """Put the descriptions back in order and add the line about each image.

        Args:
            results: The description for each position.
            total: How many images there are.
            exif_by_index: The extra info for each image, by position.

        Returns:
            The descriptions in order and a status code. The code is 0 when
            it worked and 1 when a description is missing or empty.
        """
        if exif_by_index is None:
            exif_by_index = {}

        descriptions = []
        for i in range(total):
            desc = results.get(i, "")
            if not desc or not desc.strip():
                return [], 1

            exif = exif_by_index.get(i, {})
            metadata_prefix = self._build_metadata_prefix(exif)
            if metadata_prefix:
                descriptions.append(f"[Metadata: {metadata_prefix}]\n\n{desc.strip()}")
            else:
                descriptions.append(desc)
        return descriptions, 0


async def _do_describe_image(
    image_paths: Union[str, List[str]],
    context: str = "",
    batch_size: int = 3,
    verbose: bool = False,
) -> Tuple[List[str], int]:
    """Describe images, one description each.

    Args:
        image_paths: One path or a list of paths to image files.
        context: Extra context for the model.
        batch_size: How many images the model is asked about at once.
        verbose: If True, prints extra messages.

    Returns:
        One description per image and a status code. The code is 0 when it
        worked, 1 when the model failed, and 2 when an image could not be
        read.
    """
    pipeline = _ImagePipeline(batch_size=batch_size, verbose=verbose)
    return await pipeline.process(image_paths, context)


async def describe_image(
    image_paths: Union[str, List[str]], context: str = "", verbose: bool = False
) -> Tuple[str, int]:
    """Describe images, or answer a question about them.

    Args:
        image_paths: One path or a list of paths to image files.
        context: A question about the images. Empty asks for a description.
        verbose: If True, prints extra messages.

    Returns:
        The joined descriptions or the answer, and a status code. The code
        is 0 when it worked, 1 when the model failed, and 2 when an image
        could not be read.
    """
    descriptions, status = await _do_describe_image(image_paths, context, verbose=verbose)

    if status != 0:
        return "Failed to process images.", status

    if len(descriptions) == 0:
        return "No images to process.", 0

    return await answer_question("\n==================\n".join(descriptions), context), 0


async def answer_question(content: str, question: str) -> str:
    """Answer a question using only the given content.

    Args:
        content: The text of a file, such as a description or a transcription.
        question: The question. Empty gives back the content as it is.

    Returns:
        The answer, or the content when there is no question or no answer.
    """
    question = question.strip()
    if not question:
        return content
    answer, status = await reason(
        f"{_ANSWER_PROMPT}\nUser question: {question}", content, thinking=False
    )
    if status != 0 or not answer.strip():
        print(f"[describe_image.answer_question] No answer, giving back the content: {answer[:100]}", file=sys.stderr)
        return content
    return answer.strip()
