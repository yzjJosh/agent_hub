from __future__ import annotations

import asyncio
import math
import random
import sys
import warnings
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy import uint8
from scipy.ndimage import label
from torch import no_grad, set_num_threads
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


_YOLO_CONF_THRESHOLD = 0.2
_YOLO_IOU_THRESHOLD = 0.3
_MIN_AREA_RATIO = 0.005
_MIN_CONTOUR_AREA = 100
_CLICK_JITTER = 25


class _DetectionModels:
    def __init__(self) -> None:
        set_num_threads(5)
        self.models_path = Path(get_models_dir())

        self._load_yolo_detector()
        self._load_vit_model()
        self._load_vit_processor()
        self._load_seg_model()
        self._load_seg_processor()

    def _load_yolo_detector(self) -> None:
        self.yolo_model = YOLO(self.models_path / "yolo11m-seg.pt")

    def _load_vit_model(self) -> None:
        self.vit_model = CLIPModel.from_pretrained("flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K", cache_dir=self.models_path)

    def _load_vit_processor(self) -> None:
        self.vit_processor = CLIPProcessor.from_pretrained("flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K", cache_dir=self.models_path, use_fast=True)

    def _load_seg_model(self) -> None:
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=self.models_path)

    def _load_seg_processor(self) -> None:
        self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=self.models_path, use_fast=True)

    def _check_loaded(self) -> None:
        if not self.yolo_model:
            raise ValueError("YOLO detector model was not loaded")
        if not self.seg_model:
            raise ValueError("CLIPSeg segmentation model was not loaded")
        if not self.vit_model:
            raise ValueError("CLIP ViT model was not loaded")


class Detector:
    # fmt: off
    _challenge_alias: Dict[str, str] = {
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
        "chimney": "chimney", "chimneys": "chimney"
    }
    # fmt: on

    _models: _DetectionModels | None = None

    @classmethod
    def _get_models(cls) -> _DetectionModels:
        if cls._models is None:
            cls._models = _DetectionModels()
        return cls._models

    def __init__(self, optimize_click_order: bool = True) -> None:
        self.optimize_click_order = optimize_click_order
        self._yolo_detector = _YoloDetector()
        self._clip_detector = _ClipDetector()

    async def detect_async[T](
        self,
        prompt: str,
        images: T,
        area_captcha: bool | None = None,
    ) -> tuple[list[bool], list[tuple[int, int]]]:
        """Work out which captcha tiles to click, without blocking the caller.

        Args:
            prompt: What the captcha asks you to find.
            images: The whole captcha picture, or one picture per tile.
            area_captcha: True when the tiles are parts of one big picture.
                It is read from the prompt when left out.

        Returns:
            Which tiles match, and where to click for each match. Both are
            empty when the prompt asks for something unknown, or when the
            pictures cannot be used.
        """
        return await asyncio.to_thread(self.detect, prompt, images, area_captcha=area_captcha)

    def detect[T](
        self,
        prompt: str,
        images: T,
        area_captcha: bool | None = None,
    ) -> tuple[list[bool], list[tuple[int, int]]]:
        """Work out which captcha tiles to click.

        This blocks, so callers must run it off the event loop.

        Args:
            prompt: What the captcha asks you to find.
            images: The whole captcha picture, or one picture per tile.
            area_captcha: True when the tiles are parts of one big picture.
                It is read from the prompt when left out.

        Returns:
            Which tiles match, and where to click for each match. Both are
            empty when the prompt asks for something unknown, or when the
            pictures cannot be used.
        """
        self._get_models()._check_loaded()

        response = []
        coordinates: list[tuple[int, int]] = []
        area_captcha = "square" in prompt if area_captcha is None else area_captcha
        label = split_prompt_message(prompt)

        if label not in self._challenge_alias:
            print(f"[Detector.detect] Types of challenges of label {label} not yet scheduled (Prompt: {prompt}).", file=sys.stderr)
            return [], []
        label = self._challenge_alias[label]

        byte_images, cv2_images, coords = self._prepare_image_inputs(images, area_captcha)
        if byte_images is None:
            return [], []

        coordinates = self._ensure_coordinates(byte_images, cv2_images, coords, area_captcha)
        response = self._run_detection(byte_images, cv2_images, label)

        good_coordinates: list[tuple[int, int]] = self._build_jittered_coords(response, coordinates)
        if self.optimize_click_order and len(good_coordinates) > 2:
            good_coordinates = find_lowest_distance(good_coordinates[0], good_coordinates[1:])

        return response, good_coordinates

    def _prepare_image_inputs[T](self, images: T, area_captcha: bool) -> Tuple[Optional[List[bytes]], Optional[List[cv2.typing.MatLike]], List[Tuple[int, int]]]:
        if isinstance(images, (PathLike, str)):
            images = Path(images)

        if isinstance(images, bytes) or isinstance(images, Path):
            byte_images, coords = handle_single_image(images, area_captcha)
            if isinstance(byte_images, list) and len(byte_images) > 1:
                return byte_images, handle_multiple_images(byte_images), coords
            return byte_images, None, coords

        if isinstance(images, list):
            if len(images) == 1:
                if isinstance(images[0], (PathLike, str)):
                    byte_images, coords = handle_single_image(Path(images[0]), area_captcha)
                else:
                    byte_images, coords = handle_single_image(images[0], area_captcha)
                if isinstance(byte_images, list) and len(byte_images) > 1:
                    return byte_images, handle_multiple_images(byte_images), coords
                return byte_images, None, coords

            byte_images = [
                Path(image).read_bytes() if isinstance(image, (Path, PathLike, str))
                else image for image in images
            ]
            if len(byte_images) not in (9, 16):
                print(f"[Detector.detect] Images amount must equal 9 or 16. Is: {len(byte_images)}", file=sys.stderr)
                return None, None, []

            return byte_images, handle_multiple_images(byte_images), []
        return None, None, []

    def _ensure_coordinates(self, byte_images: List[bytes], cv2_images: Optional[List[cv2.typing.MatLike]], coordinates: List[Tuple[int, int]], area_captcha: bool) -> List[Tuple[int, int]]:
        if not isinstance(byte_images, list) or len(byte_images) <= 1:
            return coordinates

        if any(coordinates):
            return coordinates

        src = cv2_images[0] if cv2_images else None
        if src is None:
            return coordinates

        height, width, _ = src.shape
        tiles_amount = 4 if area_captcha else 3
        return calculate_approximated_coords(width, height, tiles_amount)

    def _run_detection(self, byte_images: List[bytes], cv2_images: List[cv2.typing.MatLike], label: str) -> List[bool]:
        if not isinstance(byte_images, list) or len(byte_images) == 1:
            return []

        if label in self._yolo_detector._yolo_classes:
            cv2_image = create_image_grid(cv2_images)
            return self._yolo_detector._detect_image(cv2_image, len(byte_images), label)
        return self._clip_detector._detect_image(cv2_images, label)

    def _build_jittered_coords(self, response: List[bool], coordinates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        good = []
        for i, result in enumerate(response):
            if result:
                x, y = coordinates[i]
                good.append((x + random.randint(-_CLICK_JITTER, _CLICK_JITTER), y + random.randint(-_CLICK_JITTER, _CLICK_JITTER)))
        return good


class _YoloDetector:
    # fmt: off
    _yolo_classes: List[str] = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # fmt: on

    _yolo_alias: Dict[str, List[str]] = {
        "bicycle": ["bicycle"],
        "car": ["car", "truck"],
        "bus": ["bus", "truck"],
        "motorcycle": ["motorcycle"],
        "boat": ["boat"],
        "fire hydrant": ["fire hydrant", "parking meter"],
        "parking meter": ["fire hydrant", "parking meter"],
        "traffic light": ["traffic light"],
    }

    def _detect_image(self, image: cv2.typing.MatLike, tile_amount: int, task_type: str) -> list[bool]:
        response = [False for _ in range(tile_amount)]
        height, width, _ = image.shape
        tiles_per_row = int(math.sqrt(tile_amount))
        tile_width, tile_height = width // tiles_per_row, height // tiles_per_row

        # Ultralytics uses the GPU when one is present.
        outputs = Detector._get_models().yolo_model.predict(image, verbose=False, conf=_YOLO_CONF_THRESHOLD, iou=_YOLO_IOU_THRESHOLD, device="cpu")
        results = outputs[0]

        for result in results:
            if not result:
                raise ValueError("YOLO prediction result is empty")
            class_index = int(result.boxes.cls[0])
            if self._yolo_classes[class_index] not in self._yolo_alias[task_type]:
                continue

            masks = result.masks
            mask = masks.xy[0]
            response = calculate_segmentation_response(mask, response, tile_width, tile_height, tiles_per_row)

            if tile_amount == 16:
                coords = result.boxes.xyxy.flatten().tolist()
                points_start, point_end = coords[:2], coords[2:]
                tiles_in_bbox = get_tiles_in_bounding_box(image, tile_amount, tuple(points_start), tuple(point_end))
                response = [x or y for x, y in zip(response, tiles_in_bbox)]

        return response


class _ClipDetector:
    # fmt: off
    _plain_labels: List[str] = ["bicycle", "boat", "bus", "car", "fire hydrant", "motorcycle", "traffic light",  # YOLO TASKS
                    "bridge", "chimney", "crosswalk", "mountain", "palm tree", "stair", "tractor", "taxi"]

    _all_labels: List[str] = ["a bicycle", "a boat", "a bus", "a car", "a fire hydrant", "a motorcycle", "a traffic light",  # YOLO TASKS
                  "the front or bottom or side of a concrete or steel bridge supported by concrete pillars over a street or highway",
                  "A close-up of a chimney on a house, with rooftops and ceiling below",
                  "striped pedestrian crossing with white/yellow of a crosswalk stretching over a gray ground of a street",
                  "An californian green or grey landscape with trees or a bridge or street or road connecting two mountain slopes",
                  "A feather-like warm palm growing behind to a tiled rooftop, with a californian road or street",
                  "a stairway for pedestrians in front of a house or building leading to a walkway",
                  "a tractor or agricultural vehicle driving on a street or field",
                  "a taxi or a yellow car",
                  "a house wall",
                  "an empty street"]
    # fmt: on

    _thresholds: Dict[str, float] = {
        "bridge": 0.7285372716747225,
        "chimney": 0.7918647485226393,
        "crosswalk": 0.8879293048381806,
        "mountain": 0.5551278884819476,
        "palm tree": 0.8093279512040317,
        "stair": 0.9112694561691023,
        "tractor": 0.9385110986077537,
        "taxi": 0.7967491503432393,
    }

    _area_captcha_labels: Dict[str, str] = {
        "bridge": "A detailed perspective of a concrete bridge with cylindrical and rectangular supports spanning over a wide highway.",
        "chimney": "A close-up of a chimney on a house, with rooftops and ceiling below",
        "crosswalk": "striped pedestrian crossing with white/yellow of a crosswalk stretching over a gray ground of a street",
        "mountain": "An californian green or grey landscape with trees or a bridge or street or road connecting two mountain slopes",
        "palm tree": "A feather-like warm palm growing behind to a tiled rooftop, with a californian road or street",
        "stair": "a stairway for pedestrians in front of a house or building leading to a walkway",
        "tractor": "a tractor or agricultural vehicle",
        "taxi": "a yellow car or taxi",
    }

    def _clip_detect_vit(self, images: list[cv2.typing.MatLike], task_type: str) -> list[bool]:
        response = []
        inputs = Detector._get_models().vit_processor(text=self._all_labels, images=images, return_tensors="pt", padding=True)
        with no_grad():
            outputs = Detector._get_models().vit_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        results = probs.tolist()

        for result in results:
            task_index = self._plain_labels.index(task_type)
            prediction = result[task_index]
            choice = prediction >= (self._thresholds[task_type] - 0.2)

            response.append(choice)

        return response

    def _compute_heatmask(self, seg_model: CLIPSegForImageSegmentation, seg_processor: CLIPSegProcessor, image: cv2.typing.MatLike, text_label: str, threshold: float) -> torch.Tensor:
        inputs = seg_processor(text=text_label, images=[image], return_tensors="pt")
        with no_grad():
            outputs = seg_model(**inputs)

        heatmap = outputs.logits[0]
        heatmap = torch.sigmoid(heatmap)
        normalized_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        hotspot_mask = (normalized_heatmap > threshold).float()

        hotspot_mask_np = hotspot_mask.cpu().numpy()
        labeled_array, num_features = label(hotspot_mask_np)
        min_area = _MIN_AREA_RATIO * hotspot_mask_np.size

        cleaned_mask = np.zeros_like(hotspot_mask_np)
        for region_label in range(1, num_features + 1):
            region_area = np.sum(labeled_array == region_label)
            if region_area >= min_area:
                cleaned_mask[labeled_array == region_label] = 1

        return torch.from_numpy(cleaned_mask).float()

    def _upscale_mask_to_image(self, mask: torch.Tensor, image_shape: Tuple[int, int, int]) -> torch.Tensor:
        return (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(image_shape[0], image_shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    def _clipseg_detect_rd64(self, image: cv2.typing.MatLike, task_type: str, tiles_amount: int) -> list[bool]:
        response = [False for _ in range(tiles_amount)]
        segment_label = self._area_captcha_labels[task_type]

        threshold = self._thresholds[task_type] - 0.2
        hotspot_mask_cleaned = self._compute_heatmask(
            Detector._get_models().seg_model,
            Detector._get_models().seg_processor,
            image,
            segment_label,
            threshold,
        )

        mask_resized = self._upscale_mask_to_image(hotspot_mask_cleaned, image.shape)

        tiles_per_row = int(math.sqrt(tiles_amount))
        mask_width, mask_height = mask_resized.shape
        tile_width, tile_height = (
            mask_width // tiles_per_row,
            mask_height // tiles_per_row,
        )

        threshold_image = mask_resized.numpy().astype(uint8)
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > _MIN_CONTOUR_AREA:
                new_contours.append(contour)
                mask = contour.squeeze()
                response = calculate_segmentation_response(
                    mask,
                    response,
                    tile_width,
                    tile_height,
                    tiles_per_row,
                    threshold_image,
                )

        return response

    def _detect_image(self, images: list[cv2.typing.MatLike], task_type: str) -> list[bool]:
        if len(images) == 9:
            return self._clip_detect_vit(images, task_type)

        combined_image = create_image_grid(images)
        return self._clipseg_detect_rd64(combined_image, task_type, len(images))


__all__ = ["Detector"]