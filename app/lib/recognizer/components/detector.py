"""Detect matching captcha tiles and click coordinates."""

from __future__ import annotations

import asyncio
import math
import random
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import torch
from scipy.ndimage import label as label_connected_regions
from torch import no_grad, set_num_threads
from torch.nn.functional import interpolate
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)
from ultralytics import YOLO

from app.lib.shared.model_paths import get_models_dir

from .detection_processor import (
    calculate_approximated_coords,
    calculate_segmentation_response,
    find_lowest_distance,
    get_tiles_in_bounding_box,
)
from .image_processor import (
    create_image_grid,
    handle_multiple_images,
    handle_single_image,
)
from .prompt_handler import split_prompt_message

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


_CaptchaImage: TypeAlias = bytes | str | PathLike[str]
_CaptchaImageInput: TypeAlias = _CaptchaImage | Sequence[_CaptchaImage]

_TORCH_THREAD_COUNT = 5
_YOLO_CONFIDENCE_THRESHOLD = 0.2
_YOLO_IOU_THRESHOLD = 0.3
_MIN_AREA_RATIO = 0.005
_MIN_CONTOUR_AREA = 100
_CLICK_JITTER = 25
_CLASSIFICATION_GRID_SIZE = 3
_SPLIT_GRID_SIZE = 4
_CLASSIFICATION_TILE_COUNT = 9
_SPLIT_TILE_COUNT = 16
_VALID_TILE_COUNTS = (_CLASSIFICATION_TILE_COUNT, _SPLIT_TILE_COUNT)
_CLIP_THRESHOLD_MARGIN = 0.2
_MIN_POINTS_FOR_ORDERING = 3

_YOLO_MODEL_FILENAME = "yolo11m-seg.pt"
_CLIP_MODEL_NAME = "flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
_CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"


@dataclass(frozen=True)
class DetectionResult:
    """Matched tiles and click points for a captcha challenge."""

    tile_matches: list[bool]
    click_coordinates: list[tuple[int, int]]


@dataclass(frozen=True)
class _PreparedInputs:
    tile_bytes: list[bytes]
    tile_images: list[cv2.typing.MatLike] | None
    click_coordinates: list[tuple[int, int]]


@dataclass(frozen=True)
class _ClipClassificationPrompt:
    label: str
    prompt: str


@dataclass(frozen=True)
class _ClipTaskConfig:
    threshold: float
    segmentation_prompt: str


_CLIP_CLASSIFICATION_PROMPTS: tuple[_ClipClassificationPrompt, ...] = (
    _ClipClassificationPrompt("bicycle", "a bicycle"),
    _ClipClassificationPrompt("boat", "a boat"),
    _ClipClassificationPrompt("bus", "a bus"),
    _ClipClassificationPrompt("car", "a car"),
    _ClipClassificationPrompt("fire hydrant", "a fire hydrant"),
    _ClipClassificationPrompt("motorcycle", "a motorcycle"),
    _ClipClassificationPrompt("traffic light", "a traffic light"),
    _ClipClassificationPrompt(
        "bridge",
        "the front or bottom or side of a concrete or steel bridge supported by concrete pillars over a street or highway",
    ),
    _ClipClassificationPrompt(
        "chimney",
        "A close-up of a chimney on a house, with rooftops and ceiling below",
    ),
    _ClipClassificationPrompt(
        "crosswalk",
        "striped pedestrian crossing with white/yellow of a crosswalk stretching over a gray ground of a street",
    ),
    _ClipClassificationPrompt(
        "mountain",
        "An californian green or grey landscape with trees or a bridge or street or road connecting two mountain slopes",
    ),
    _ClipClassificationPrompt(
        "palm tree",
        "A feather-like warm palm growing behind to a tiled rooftop, with a californian road or street",
    ),
    _ClipClassificationPrompt(
        "stair",
        "a stairway for pedestrians in front of a house or building leading to a walkway",
    ),
    _ClipClassificationPrompt(
        "tractor",
        "a tractor or agricultural vehicle driving on a street or field",
    ),
    _ClipClassificationPrompt("taxi", "a taxi or a yellow car"),
)

_CLIP_NEGATIVE_PROMPTS: tuple[str, ...] = (
    "a house wall",
    "an empty street",
)

_CLIP_TEXT_PROMPTS: list[str] = [
    *(entry.prompt for entry in _CLIP_CLASSIFICATION_PROMPTS),
    *_CLIP_NEGATIVE_PROMPTS,
]

_CLIP_LABEL_INDEX: dict[str, int] = {
    entry.label: index for index, entry in enumerate(_CLIP_CLASSIFICATION_PROMPTS)
}

_CLIP_TASK_CONFIGS: dict[str, _ClipTaskConfig] = {
    "bridge": _ClipTaskConfig(
        threshold=0.7285372716747225,
        segmentation_prompt="A detailed perspective of a concrete bridge with cylindrical and rectangular supports spanning over a wide highway.",
    ),
    "chimney": _ClipTaskConfig(
        threshold=0.7918647485226393,
        segmentation_prompt="A close-up of a chimney on a house, with rooftops and ceiling below",
    ),
    "crosswalk": _ClipTaskConfig(
        threshold=0.8879293048381806,
        segmentation_prompt="striped pedestrian crossing with white/yellow of a crosswalk stretching over a gray ground of a street",
    ),
    "mountain": _ClipTaskConfig(
        threshold=0.5551278884819476,
        segmentation_prompt="An californian green or grey landscape with trees or a bridge or street or road connecting two mountain slopes",
    ),
    "palm tree": _ClipTaskConfig(
        threshold=0.8093279512040317,
        segmentation_prompt="A feather-like warm palm growing behind to a tiled rooftop, with a californian road or street",
    ),
    "stair": _ClipTaskConfig(
        threshold=0.9112694561691023,
        segmentation_prompt="a stairway for pedestrians in front of a house or building leading to a walkway",
    ),
    "tractor": _ClipTaskConfig(
        threshold=0.9385110986077537,
        segmentation_prompt="a tractor or agricultural vehicle",
    ),
    "taxi": _ClipTaskConfig(
        threshold=0.7967491503432393,
        segmentation_prompt="a yellow car or taxi",
    ),
}


class _DetectionModels:
    """Loaded vision models shared across detector runs."""

    def __init__(self) -> None:
        set_num_threads(_TORCH_THREAD_COUNT)
        models_path = Path(get_models_dir())
        self.yolo_model = YOLO(models_path / _YOLO_MODEL_FILENAME)
        self.clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME, cache_dir=models_path)
        self.clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME, cache_dir=models_path, use_fast=True)
        self.segmentation_model = CLIPSegForImageSegmentation.from_pretrained(
            _CLIPSEG_MODEL_NAME, cache_dir=models_path
        )
        self.segmentation_processor = CLIPSegProcessor.from_pretrained(
            _CLIPSEG_MODEL_NAME, cache_dir=models_path, use_fast=True
        )

    def check_loaded(self) -> None:
        if not self.yolo_model:
            raise ValueError("YOLO detector model was not loaded")
        if not self.segmentation_model:
            raise ValueError("CLIPSeg segmentation model was not loaded")
        if not self.clip_model:
            raise ValueError("CLIP ViT model was not loaded")


class Detector:
    """Find matching captcha tiles and compute click points for them."""

    # fmt: off
    _challenge_alias: dict[str, str] = {
        "car": "car", "cars": "car", "vehicles": "car",
        "taxis": "taxi", "taxi": "taxi",
        "bus": "bus", "buses": "bus",
        "motorcycle": "motorcycle", "motorcycles": "motorcycle",
        "bicycle": "bicycle", "bicycles": "bicycle",
        "boats": "boat", "boat": "boat",
        "tractors": "tractor", "tractor": "tractor",
        "stairs": "stair", "stair": "stair",
        "palm trees": "palm tree", "palm tree": "palm tree",
        "fire hydrants": "fire hydrant", "a fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
        "parking meters": "parking meter", "parking meter": "parking meter",
        "crosswalks": "crosswalk", "crosswalk": "crosswalk",
        "traffic lights": "traffic light", "traffic light": "traffic light",
        "bridges": "bridge", "bridge": "bridge",
        "mountains or hills": "mountain", "mountain or hill": "mountain", "mountain": "mountain", "mountains": "mountain", "hills": "mountain", "hill": "mountain",
        "chimney": "chimney", "chimneys": "chimney",
    }
    # fmt: on

    _models: _DetectionModels | None = None

    @classmethod
    def _get_models(cls) -> _DetectionModels:
        if cls._models is None:
            cls._models = _DetectionModels()
        return cls._models

    def __init__(self, optimize_click_order: bool = True) -> None:
        """Create a captcha tile detector.

        Args:
            optimize_click_order: True to sort click points into a short path.
        """
        self._should_optimize_click_order = optimize_click_order
        self._yolo_detector = _YoloDetector()
        self._clip_detector = _ClipDetector()

    async def detect_async(
        self,
        prompt: str,
        images: _CaptchaImageInput,
        is_split_image: bool | None = None,
    ) -> DetectionResult:
        """Work out which captcha tiles to click, without blocking the caller.

        Args:
            prompt: What the captcha asks you to find.
            images: The whole captcha picture, or one picture per tile.
            is_split_image: True when the tiles are parts of one big picture.
                It is read from the prompt when left out.

        Returns:
            Matched tiles and click points. Both are empty when the prompt asks
            for something unknown, or when the pictures cannot be used.
        """
        return await asyncio.to_thread(self.detect, prompt, images, is_split_image=is_split_image)

    def detect(
        self,
        prompt: str,
        images: _CaptchaImageInput,
        is_split_image: bool | None = None,
    ) -> DetectionResult:
        """Work out which captcha tiles to click.

        This blocks, so callers must run it off the event loop.

        Args:
            prompt: What the captcha asks you to find.
            images: The whole captcha picture, or one picture per tile.
            is_split_image: True when the tiles are parts of one big picture.
                It is read from the prompt when left out.

        Returns:
            Matched tiles and click points. Both are empty when the prompt asks
            for something unknown, or when the pictures cannot be used.
        """
        models = self._get_models()
        models.check_loaded()

        is_split_grid = "square" in prompt if is_split_image is None else is_split_image
        task_label = split_prompt_message(prompt)

        if task_label not in self._challenge_alias:
            print(
                f"[Detector.detect] Types of challenges of label {task_label} not yet scheduled (Prompt: {prompt}).",
                file=sys.stderr,
            )
            return DetectionResult([], [])
        task_label = self._challenge_alias[task_label]

        prepared = self._prepare_image_inputs(images, is_split_grid)
        if prepared is None:
            return DetectionResult([], [])

        tile_coordinates = self._fill_missing_coordinates(prepared, is_split_grid)
        tile_matches = self._run_detection(prepared, task_label, models)

        click_coordinates = self._build_jittered_coordinates(tile_matches, tile_coordinates)
        if self._should_optimize_click_order and len(click_coordinates) >= _MIN_POINTS_FOR_ORDERING:
            click_coordinates = find_lowest_distance(click_coordinates[0], click_coordinates[1:])

        return DetectionResult(tile_matches=tile_matches, click_coordinates=click_coordinates)

    def _prepare_single_image(self, image: _CaptchaImage, is_split_image: bool) -> _PreparedInputs:
        single_image: bytes | Path = Path(image) if isinstance(image, (PathLike, str)) else image
        tile_bytes, click_coordinates = handle_single_image(single_image, is_split_image)
        if len(tile_bytes) > 1:
            return _PreparedInputs(tile_bytes, handle_multiple_images(tile_bytes), click_coordinates)
        return _PreparedInputs(tile_bytes, None, click_coordinates)

    def _prepare_image_inputs(
        self,
        images: _CaptchaImageInput,
        is_split_image: bool,
    ) -> _PreparedInputs | None:
        if isinstance(images, (bytes, str, PathLike)):
            return self._prepare_single_image(images, is_split_image)

        if not isinstance(images, Sequence) or isinstance(images, (bytes, bytearray, memoryview)):
            return None

        if len(images) == 1:
            return self._prepare_single_image(images[0], is_split_image)

        tile_bytes = [
            Path(image).read_bytes() if isinstance(image, (PathLike, str)) else image
            for image in images
        ]
        if len(tile_bytes) not in _VALID_TILE_COUNTS:
            print(
                f"[Detector.detect] Images amount must equal 9 or 16. Is: {len(tile_bytes)}",
                file=sys.stderr,
            )
            return None

        return _PreparedInputs(tile_bytes, handle_multiple_images(tile_bytes), [])

    def _fill_missing_coordinates(
        self,
        prepared: _PreparedInputs,
        is_split_image: bool,
    ) -> list[tuple[int, int]]:
        if len(prepared.tile_bytes) <= 1 or any(prepared.click_coordinates):
            return prepared.click_coordinates

        first_tile = prepared.tile_images[0] if prepared.tile_images else None
        if first_tile is None:
            return prepared.click_coordinates

        height, width, _ = first_tile.shape
        tiles_per_row = _SPLIT_GRID_SIZE if is_split_image else _CLASSIFICATION_GRID_SIZE
        return calculate_approximated_coords(width, height, tiles_per_row)

    def _run_detection(
        self,
        prepared: _PreparedInputs,
        task_label: str,
        models: _DetectionModels,
    ) -> list[bool]:
        if len(prepared.tile_bytes) <= 1 or prepared.tile_images is None:
            return []

        if self._yolo_detector.can_detect_label(task_label):
            grid_image = create_image_grid(prepared.tile_images)
            return self._yolo_detector.detect_tiles(grid_image, len(prepared.tile_bytes), task_label, models)
        return self._clip_detector.detect_tiles(prepared.tile_images, task_label, models)

    def _build_jittered_coordinates(
        self,
        tile_matches: list[bool],
        coordinates: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        jittered_coordinates: list[tuple[int, int]] = []
        for index, is_match in enumerate(tile_matches):
            if is_match:
                center_x, center_y = coordinates[index]
                jittered_coordinates.append(
                    (
                        center_x + random.randint(-_CLICK_JITTER, _CLICK_JITTER),
                        center_y + random.randint(-_CLICK_JITTER, _CLICK_JITTER),
                    )
                )
        return jittered_coordinates


class _YoloDetector:
    """Detect common street objects across a tiled captcha grid."""

    # fmt: off
    _yolo_classes: tuple[str, ...] = (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    )
    # fmt: on

    _yolo_alias: dict[str, tuple[str, ...]] = {
        "bicycle": ("bicycle",),
        "car": ("car", "truck"),
        "bus": ("bus", "truck"),
        "motorcycle": ("motorcycle",),
        "boat": ("boat",),
        "fire hydrant": ("fire hydrant", "parking meter"),
        "parking meter": ("fire hydrant", "parking meter"),
        "traffic light": ("traffic light",),
    }

    def can_detect_label(self, task_label: str) -> bool:
        return task_label in self._yolo_alias

    def detect_tiles(
        self,
        image: cv2.typing.MatLike,
        tile_count: int,
        task_label: str,
        models: _DetectionModels,
    ) -> list[bool]:
        """Mark every grid tile containing an object matching the challenge label."""
        tile_matches = [False for _ in range(tile_count)]
        height, width, _ = image.shape
        tiles_per_row = int(math.sqrt(tile_count))
        tile_width, tile_height = width // tiles_per_row, height // tiles_per_row

        # Ultralytics uses the GPU when one is present.
        outputs = models.yolo_model.predict(
            image,
            verbose=False,
            conf=_YOLO_CONFIDENCE_THRESHOLD,
            iou=_YOLO_IOU_THRESHOLD,
            device="cpu",
        )
        results = outputs[0]

        for result in results:  # type: ignore[attr-defined]
            if not result:
                raise ValueError("YOLO prediction result is empty")
            class_index = int(result.boxes.cls[0])
            if self._yolo_classes[class_index] not in self._yolo_alias[task_label]:
                continue

            mask_points = result.masks.xy[0]
            tile_matches = calculate_segmentation_response(
                mask_points,
                tile_matches,
                tile_width,
                tile_height,
                tiles_per_row,
            )

            if tile_count == _SPLIT_TILE_COUNT:
                box_coordinates = result.boxes.xyxy.flatten().tolist()
                box_start = (box_coordinates[0], box_coordinates[1])
                box_end = (box_coordinates[2], box_coordinates[3])
                tiles_in_box = get_tiles_in_bounding_box(image, tile_count, box_start, box_end)
                tile_matches = [is_matched or is_in_box for is_matched, is_in_box in zip(tile_matches, tiles_in_box)]

        return tile_matches


class _ClipDetector:
    """Detect matching tiles for challenge labels outside object detection."""

    def _classify_tiles(
        self,
        images: list[cv2.typing.MatLike],
        task_label: str,
        models: _DetectionModels,
    ) -> list[bool]:
        inputs = models.clip_processor(
            text=_CLIP_TEXT_PROMPTS,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        with no_grad():
            outputs = models.clip_model(**inputs)

        tile_probabilities = outputs.logits_per_image.softmax(dim=1).tolist()
        task_index = _CLIP_LABEL_INDEX[task_label]
        threshold = _CLIP_TASK_CONFIGS[task_label].threshold - _CLIP_THRESHOLD_MARGIN
        return [probabilities[task_index] >= threshold for probabilities in tile_probabilities]

    def _build_region_mask(
        self,
        segmentation_model: CLIPSegForImageSegmentation,
        segmentation_processor: CLIPSegProcessor,
        image: cv2.typing.MatLike,
        text_prompt: str,
        threshold: float,
    ) -> torch.Tensor:
        """Build a binary region mask for the target object in a grid image."""
        inputs = segmentation_processor(text=text_prompt, images=[image], return_tensors="pt")
        with no_grad():
            outputs = segmentation_model(**inputs)

        heatmap = torch.sigmoid(outputs.logits[0])
        normalized_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        hotspot_mask = (normalized_heatmap > threshold).float()

        hotspot_mask_array = hotspot_mask.cpu().numpy()
        labeled_regions, region_count = label_connected_regions(hotspot_mask_array)
        min_area = _MIN_AREA_RATIO * hotspot_mask_array.size

        cleaned_mask = np.zeros_like(hotspot_mask_array)
        for region_id in range(1, region_count + 1):
            region_area = np.sum(labeled_regions == region_id)
            if region_area >= min_area:
                cleaned_mask[labeled_regions == region_id] = 1

        return torch.from_numpy(cleaned_mask).float()

    def _upscale_mask_to_image(
        self,
        mask: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        return (
            interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    def _detect_grid_regions(
        self,
        image: cv2.typing.MatLike,
        task_label: str,
        tile_count: int,
        models: _DetectionModels,
    ) -> list[bool]:
        tile_matches = [False for _ in range(tile_count)]
        task_config = _CLIP_TASK_CONFIGS[task_label]
        threshold = task_config.threshold - _CLIP_THRESHOLD_MARGIN

        region_mask = self._build_region_mask(
            models.segmentation_model,
            models.segmentation_processor,
            image,
            task_config.segmentation_prompt,
            threshold,
        )
        image_height, image_width = image.shape[0], image.shape[1]
        resized_mask = self._upscale_mask_to_image(region_mask, image_height, image_width)

        tiles_per_row = int(math.sqrt(tile_count))
        mask_height, mask_width = resized_mask.shape
        tile_width = mask_width // tiles_per_row
        tile_height = mask_height // tiles_per_row

        threshold_image = resized_mask.numpy().astype(np.uint8)
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > _MIN_CONTOUR_AREA:
                tile_matches = calculate_segmentation_response(
                    contour.squeeze(),
                    tile_matches,
                    tile_width,
                    tile_height,
                    tiles_per_row,
                    threshold_image,
                )

        return tile_matches

    def detect_tiles(
        self,
        images: list[cv2.typing.MatLike],
        task_label: str,
        models: _DetectionModels,
    ) -> list[bool]:
        """Mark every matching tile in the given grid of tile pictures."""
        if len(images) == _CLASSIFICATION_TILE_COUNT:
            return self._classify_tiles(images, task_label, models)

        combined_image = create_image_grid(images)
        return self._detect_grid_regions(combined_image, task_label, len(images), models)


__all__ = ["DetectionResult", "Detector"]
