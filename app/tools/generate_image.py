"""Make images with the local ComfyUI models.

The finished file is saved in the resource.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from typing import ClassVar, override

from app.lib.comfyui.client import ComfyUIClient
from app.lib.comfyui.media import random_seed, save_first_file
from app.lib.comfyui.workflows import build_t2i_prompt
from app.lib.shared.messages import saved_file_message
from app.lib.shared.paths import resources_dir
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["generate_image_tool"]


class _GenerateImageTool(Tool):
    name: ClassVar[str] = "generate_image"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    result_render_type: ClassVar[str] = "text"
    description: ClassVar[str] = (
        "Generate a brand-new image from a text prompt with the local "
        "ComfyUI models, from the prompt alone with no reference images. Use "
        "for pure text to image, in contrast to edit_image which builds "
        "images from reference images."
    )
    summary: ClassVar[str] = "Generate an image via ComfyUI"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "prompt",
            str,
            kind="longtext",
            description="Text prompt describing the image to generate.",
        ),
        Param(
            "aspect_ratio",
            str,
            "16:9",
            enum=("16:9", "9:16", "4:3", "3:4", "1:1"),
            description="Output aspect ratio.",
        ),
        Param(
            "upscale",
            bool,
            False,
            description="Apply the upscaler to the output, otherwise keep the base resolution.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        prompt = params["prompt"]
        if not prompt or not prompt.strip():
            raise ToolMessageError("prompt must be a non-empty string.")
        aspect_ratio = params.get("aspect_ratio", "16:9")
        upscale = params.get("upscale", False)

        api = build_t2i_prompt(prompt, random_seed(), aspect_ratio, upscale)
        result = await ComfyUIClient().generate(api)

        rel = await save_first_file(result, self.name, "png")
        return ToolResult(text=saved_file_message("image", os.path.join(resources_dir(), rel)), files=(rel,))

    @override
    async def cache_key(self, params: dict) -> str:
        payload: dict = {
            "prompt": params["prompt"],
            "aspect_ratio": params.get("aspect_ratio", "16:9"),
            "upscale": params.get("upscale", False),
            "pipeline": "qwen-image-2.1",
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()


generate_image_tool = _GenerateImageTool()
