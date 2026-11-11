"""Make short videos with the local ComfyUI models.

The frames come from the resource, and the finished file is saved there too.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from typing import ClassVar, override

from app.lib.comfyui.client import ComfyUIClient
from app.lib.comfyui.media import random_seed, reference_download_url, reference_fingerprints, save_first_file
from app.lib.comfyui.workflows import build_f2lv_prompt, build_i2v_prompt, build_t2v_prompt
from app.lib.shared.messages import saved_file_message
from app.lib.shared.paths import resources_dir
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["generate_video_tool"]


def _pipeline_for(first_frame: str | None, last_frame: str | None) -> str:
    """The pipeline the given frames pick."""
    if first_frame and last_frame:
        return "f2lv"
    return "i2v" if first_frame else "t2v"


class _GenerateVideoTool(Tool):
    name: ClassVar[str] = "generate_video"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    result_render_type: ClassVar[str] = "text"
    description: ClassVar[str] = (
        "Generate a short video from a text prompt with the local ComfyUI models. "
        "The generated video can include audio such as background music and speech. "
        "Optionally the video starts on a first frame and ends on a last frame image "
        "from the resource. Use for AI-generated short videos, in contrast to "
        "generate_image which produces still pictures."
    )
    summary: ClassVar[str] = "Generate a short video via ComfyUI"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "prompt",
            str,
            kind="longtext",
            description=(
                "Text prompt describing the video to generate, or the motion between "
                "the two frames when both are given."
            ),
        ),
        Param(
            "first_frame",
            str | None,
            None,
            kind="resource",
            description="Image in the resource to use as the video's first frame.",
        ),
        Param(
            "last_frame",
            str | None,
            None,
            kind="resource",
            description=(
                "Image in the resource to use as the video's last frame. It requires "
                "first_frame."
            ),
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
        first_frame = params.get("first_frame")
        last_frame = params.get("last_frame")
        aspect_ratio = params.get("aspect_ratio", "16:9")
        upscale = params.get("upscale", False)

        if last_frame and not first_frame:
            raise ValueError("last_frame requires first_frame.")
        pipeline = _pipeline_for(first_frame, last_frame)
        first_url = await reference_download_url(first_frame) if first_frame else None
        last_url = await reference_download_url(last_frame) if last_frame else None

        seed = random_seed()
        if pipeline == "f2lv":
            api = build_f2lv_prompt(prompt, first_url, last_url, seed, aspect_ratio, upscale)
        elif pipeline == "i2v":
            api = build_i2v_prompt(prompt, first_url, seed, aspect_ratio, upscale)
        else:
            api = build_t2v_prompt(prompt, seed, aspect_ratio, upscale)
        result = await ComfyUIClient().generate(api)

        rel = await save_first_file(result, self.name, "mp4")
        return ToolResult(text=saved_file_message("video", os.path.join(resources_dir(), rel)), files=(rel,))

    @override
    async def cache_key(self, params: dict) -> str:
        payload = json.dumps(
            {
                "prompt": params["prompt"],
                "aspect_ratio": params.get("aspect_ratio", "16:9"),
                "upscale": params.get("upscale", False),
                "first_frame": await reference_fingerprints([params["first_frame"]]) if params.get("first_frame") else None,
                "last_frame": await reference_fingerprints([params["last_frame"]]) if params.get("last_frame") else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


generate_video_tool = _GenerateVideoTool()
