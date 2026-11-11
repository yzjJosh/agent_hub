"""Edit one image with the local ComfyUI models.

The image comes from the resource, and the finished file is saved there too.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from typing import ClassVar, override

from app.lib.comfyui.client import ComfyUIClient
from app.lib.comfyui.media import (
    fit_to_box,
    image_dimensions,
    random_seed,
    reference_download_url,
    reference_fingerprints,
    save_first_file,
)
from app.lib.comfyui.workflows import build_edit_prompt
from app.lib.shared.messages import saved_file_message
from app.lib.shared.paths import resources_dir
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["edit_image_tool"]

_MAX_REFERENCES = 2


class _EditImageTool(Tool):
    name: ClassVar[str] = "edit_image"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    result_render_type: ClassVar[str] = "text"
    description: ClassVar[str] = (
        "Edit an image from the resource with a text prompt using the local "
        "ComfyUI models, and build new imagery from reference images. The "
        "output follows the input image, and this is the only local ComfyUI "
        "image tool that accepts reference images. Use to modify a given "
        "image or to generate imagery from reference images, in contrast to "
        "generate_image which paints a brand-new image from a text prompt alone."
    )
    summary: ClassVar[str] = "Edit an image via ComfyUI"
    params: ClassVar[tuple[Param, ...]] = (
        Param("image", str, kind="resource", description="The image in the resource to edit."),
        Param("prompt", str, kind="longtext", description="The edit to apply to the image."),
        Param(
            "reference_images",
            list[str] | None,
            None,
            kind="resource_list",
            description=(
                f"Reference images in the resource that guide the editing, up to {_MAX_REFERENCES}. "
                "The image param remains the canvas the output follows. Use when the edit "
                "should take subjects or detail from other images in the resource."
            ),
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        image = params["image"]
        prompt = params["prompt"]
        if not prompt or not prompt.strip():
            raise ToolMessageError("prompt must be a non-empty string.")
        references = params.get("reference_images") or []

        if len(references) > _MAX_REFERENCES:
            raise ValueError(
                f"image editing takes at most {_MAX_REFERENCES} reference images, "
                f"got {len(references)}"
            )

        url = await reference_download_url(image)
        urls = [await reference_download_url(reference) for reference in references]
        canvas_dims = await image_dimensions(image)
        reference_dims = [await image_dimensions(reference) for reference in references]
        seed = random_seed()
        result = await ComfyUIClient().generate(
            build_edit_prompt(
                prompt,
                url,
                seed,
                image_dims=fit_to_box(canvas_dims[0], canvas_dims[1]),
                extra_images=urls,
                extra_image_dims=[fit_to_box(width, height) for width, height in reference_dims],
            )
        )

        rel = await save_first_file(result, self.name, "png")
        return ToolResult(text=saved_file_message("image", os.path.join(resources_dir(), rel)), files=(rel,))

    @override
    async def cache_key(self, params: dict) -> str:
        references = params.get("reference_images") or []
        payload = json.dumps(
            {
                "prompt": params["prompt"],
                "image": await reference_fingerprints([params["image"]]),
                "reference_images": await reference_fingerprints(references),
                "pipeline": "qwen-image-2.1",
                "input_scale": "fit2048",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


edit_image_tool = _EditImageTool()
