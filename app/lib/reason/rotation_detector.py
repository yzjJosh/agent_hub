"""Find which way an image is turned and turn it upright.

The model sorts an image into one of four classes. Class 0 is already upright,
class 1 needs a quarter turn clockwise, class 2 needs a half turn, and class 3
needs a quarter turn counter clockwise.
"""

__all__ = ["rotate_images_in_place"]

import asyncio
import os
import sys
from typing import List, Optional
from PIL import Image, ImageOps
import aiofiles
import httpx
import numpy as np
import onnxruntime as ort

from app.lib.shared.model_paths import get_models_dir, get_model_path
from .onnx_inference import run_inference, softmax

_MODEL_DIR = get_models_dir()
_MODEL_PATH = get_model_path("orientation_model.onnx")

# The model is loaded once and kept here.
_SESSION: Optional[ort.InferenceSession] = None

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The model only takes images this many pixels wide and tall.
_IMAGE_SIZE = 384

# How many degrees counter clockwise each class must be turned.
_CLASS_TO_ROTATION = {
    0: 0,
    1: 270,
    2: 180,
    3: 90,
}

_CLASS_NAMES = {
    0: "upright (0°)",
    1: "needs 90° CW rotation",
    2: "needs 180° rotation",
    3: "needs 90° CCW rotation",
}

_DOWNLOAD_URL = (
    "https://github.com/duartebarbosadev/deep-image-orientation-detection/"
    "releases/download/v2/orientation_model_v2_0.9882.onnx"
)


async def _ensure_model() -> None:
    """Put the orientation model file on disk when it is not there yet.

    A download that fails is reported and leaves no file behind.
    """
    if _SESSION is not None or await asyncio.to_thread(os.path.exists, _MODEL_PATH):
        return

    try:
        await _download_model()
    except Exception as e:
        print(f"Error downloading orientation model: {e}", file=sys.stderr)


async def _download_model() -> None:
    """Download the orientation model file.

    Raises:
        httpx.HTTPError: If the download fails.
        OSError: If the file cannot be written.
    """
    await asyncio.to_thread(os.makedirs, _MODEL_DIR, exist_ok=True)
    print(f"Downloading orientation model to {_MODEL_PATH}...", file=sys.stderr)
    partial_path = f"{_MODEL_PATH}.part"
    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", _DOWNLOAD_URL) as response:
                response.raise_for_status()
                async with aiofiles.open(partial_path, "wb") as fh:
                    async for chunk in response.aiter_bytes():
                        await fh.write(chunk)
    except Exception:
        await asyncio.to_thread(_remove_partial, partial_path)
        raise
    await asyncio.to_thread(os.replace, partial_path, _MODEL_PATH)


def _remove_partial(path: str) -> None:
    """Delete a half written download.

    Args:
        path: The file to delete.
    """
    if os.path.exists(path):
        os.remove(path)


def _get_session() -> ort.InferenceSession:
    """Get the orientation model, loading it the first time.

    Returns:
        The loaded model.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    _SESSION = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _SESSION


def _preprocess_image(img: Image.Image) -> np.ndarray:
    """Get an image ready for the orientation model.

    The image must already be in RGB.

    Args:
        img: The image to get ready.

    Returns:
        Image data shaped 1 by 3 by 384 by 384.
    """
    img_resized = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = (img_array - _IMAGENET_MEAN) / _IMAGENET_STD
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)

def _get_image_rotation(image_path: str, verbose: bool = False) -> int:
    """Find how far counter clockwise an image must turn to be upright.

    Args:
        image_path: Path to the image file.
        verbose: If True, prints what the model picked.

    Returns:
        The turn in degrees counter clockwise, one of 0, 90, 180, or 270.
        The answer is 0 when the model is unsure or something goes wrong.
    """
    try:
        session = _get_session()

        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = _preprocess_image(img)
            logits = run_inference(session, img_array)

            predicted_class = int(np.argmax(logits))
            probs = softmax(logits)

            if verbose:
                _print_rotation_details(image_path, predicted_class, probs)

            if probs[predicted_class] < 0.7:
                if verbose:
                    print(f"Predicted score too low ({probs[predicted_class]:.2%}). Fallback to upright.")
                predicted_class = 0

            return _CLASS_TO_ROTATION[predicted_class]

    except Exception as e:
        print(f"Error detecting rotation for {image_path}: {e}", file=sys.stderr)
        return 0


def _print_rotation_details(image_path: str, predicted_class: int, probs: np.ndarray) -> None:
    """Print what the model picked for one image.

    Args:
        image_path: Path to the image file.
        predicted_class: The class the model picked.
        probs: One probability per class.
    """
    print(f"Image: {image_path}")
    print(f"  Predicted: {_CLASS_NAMES[predicted_class]} (confidence: {probs[predicted_class]:.2%})")
    for i, prob in enumerate(probs):
        print(f"  {_CLASS_NAMES[i]}: {prob:.2%}")

async def rotate_images_in_place(image_paths: List[str], verbose: bool = False) -> None:
    """Turn images upright and save them back over the original files.

    An image that cannot be read or turned is left alone. An empty list
    returns without reaching for the model.

    Args:
        image_paths: List of file paths to images.
        verbose: If True, prints progress messages.
    """
    if not image_paths:
        return

    await _ensure_model()
    await asyncio.to_thread(_rotate_all, image_paths, verbose)


def _rotate_all(image_paths: List[str], verbose: bool) -> None:
    """Turn every image upright and write it back.

    Args:
        image_paths: List of file paths to images.
        verbose: If True, prints progress messages.
    """
    for path in image_paths:
        rotation = _get_image_rotation(path, verbose=verbose)
        
        if rotation == 0:
            if verbose:
                print(f"Image '{path}' is already upright.")
            continue
            
        try:
            if verbose:
                print(f"Rotating '{path}' by {rotation}° counter-clockwise...")
            
            with Image.open(path) as img:
                # The stored orientation is baked in before the turn.
                img = ImageOps.exif_transpose(img)
                
                # A positive angle turns the image counter clockwise.
                rotated_img = img.rotate(rotation, expand=True)
                rotated_img.save(path)
                
            if verbose:
                print(f"Successfully rotated '{path}'.")
                
        except Exception as e:
            print(f"Error rotating image '{path}': {e}", file=sys.stderr)

