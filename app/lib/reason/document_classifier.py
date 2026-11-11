"""Tell whether an image is a document or a plain picture.

A document is something like a form, a certificate, or a page of text. A
picture is anything else, such as a photo or a screenshot.
"""

__all__ = ["is_document"]

import asyncio
import os
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import onnxruntime as ort

from app.lib.shared.model_paths import get_model_path
from .onnx_inference import run_inference, softmax

# The model is loaded once and kept here.
_SESSION: Optional[ort.InferenceSession] = None
_MODEL_PATH = get_model_path("document_classifier.onnx")

# The model only takes images this many pixels wide and tall.
_IMAGE_SIZE = 224

# The name the model gives each class, in class order.
_LABELS = ["picture", "document"]


def _get_session() -> ort.InferenceSession:
    """Get the classifier model, loading it the first time.

    Returns:
        The loaded model.

    Raises:
        ImportError: If the model runner is missing.
        FileNotFoundError: If the model file is missing.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    if ort is None:
        raise ImportError(
            "onnxruntime not installed. Please install it with: pip install onnxruntime"
        )

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. Download onnx/model_quantized.onnx "
            f"from the Hugging Face repository Mozilla/docornot and save it as {_MODEL_PATH}."
        )

    _SESSION = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _SESSION


def _preprocess_image(img: Image.Image) -> np.ndarray:
    """Get an image ready for the classifier model.

    The image must already be in RGB.

    Args:
        img: The image to get ready.

    Returns:
        Image data shaped 1 by 3 by 224 by 224.
    """
    img_resized = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)


def _predict_class(logits: np.ndarray) -> Tuple[str, str, float]:
    """Read the picked class out of the raw scores.

    Args:
        logits: One raw score per class.

    Returns:
        The group name, the model class name, and how sure the model is.
    """
    predicted_class = int(np.argmax(logits))
    probs = softmax(logits)
    confidence = float(probs[predicted_class])
    raw_class = _LABELS[predicted_class]
    category = "document" if raw_class == "document" else "non_document"
    return category, raw_class, confidence


def _classify_document(image_path: str) -> Tuple[str, str, float]:
    """Sort an image into a document or a picture.

    Args:
        image_path: Path to the image file.

    Returns:
        The group name, either document or non_document, the model class name,
        either document or picture, and how sure the model is from 0 to 1.

    Raises:
        FileNotFoundError: If the image file or the model file is missing.
        ValueError: If the path is empty or the image cannot be read.
        ImportError: If the model runner is missing.
    """
    if not image_path:
        raise ValueError("Image path cannot be empty")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    session = _get_session()

    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = _preprocess_image(img)
            logits = run_inference(session, img_array)
            return _predict_class(logits)

    except Exception as e:
        raise ValueError(f"Failed to process image {image_path}: {e}") from e


async def is_document(path: str, verbose: bool = False) -> bool:
    """Check whether an image is a document.

    Args:
        path: Path to the image file.
        verbose: If True, prints what the model picked.

    Returns:
        True if the image is a document, False if it is not or something
        goes wrong.
    """
    return await asyncio.to_thread(_is_document, path, verbose)


def _is_document(path: str, verbose: bool) -> bool:
    """Run the classifier over one image file.

    Args:
        path: Path to the image file.
        verbose: If True, prints what the model picked.

    Returns:
        True if the image is a document, False if it is not or something
        goes wrong.
    """
    try:
        category, _, confidence = _classify_document(path)
        if verbose:
            print(f"{category} ({confidence:.1%})")
        return category == "document"
    except Exception:
        return False


