"""Tests for app.tools.generate_image."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.tools.generate_image import generate_image_tool
from tests.app.conftest import (
    assert_saved_image,
    comfyui_online,
    live_image_run,
    point_resources_at,
    run_media_tool,
    submitted_prompt,
    wire_media_client,
)
from tests.app.lib.tool.conftest import run_text

_PNG_BYTES = b"\x89PNG\r\n\x1agenerated image bytes"
_PNG_FILENAME = "media_gen_t2i/out_00001_.png"
_CLIENT_PATH = "app.tools.generate_image.ComfyUIClient"

_EXP_T2I_16_9_W = 2048
_EXP_T2I_16_9_H = 1152
_EXP_T2I_16_9_2X_W = 4096
_EXP_T2I_16_9_2X_H = 2304
_EXP_T2I_4_3_W = 1792
_EXP_T2I_4_3_H = 1344

# Each model loader input of the t2i prompt, with its variable and default
# filename.
_T2I_MODEL_INPUTS = [
    ("1", "unet_name", "COMFYUI_IMAGE_UNET", "qwen_image_2.1_int8_convrot.safetensors"),
    ("3", "clip_name", "COMFYUI_IMAGE_CLIP", "qwen3vl_8b_w4a8.safetensors"),
    ("4", "vae_name", "COMFYUI_IMAGE_VAE", "qwen_image_2.1_vae_bf16.safetensors"),
    ("23", "model_name", "COMFYUI_UPSCALE_MODEL", "RealESRGAN_x2plus.pth"),
]
_CONFIGURED_MODEL = "configured_model.safetensors"


class TestGenerateImageShape:
    """The tool declares the image surface and nothing else."""

    def test_tool_shape(self) -> None:
        assert generate_image_tool.name == "generate_image"
        assert generate_image_tool.is_slow is True
        assert generate_image_tool.staleness == timedelta(hours=24)
        assert generate_image_tool.result_render_type == "text"
        assert generate_image_tool.summary
        assert "\n" not in generate_image_tool.summary
        assert len(generate_image_tool.summary) <= 80

    def test_params(self) -> None:
        params = {p.name: p for p in generate_image_tool.params}
        assert [p.name for p in generate_image_tool.params] == [
            "prompt",
            "aspect_ratio",
            "upscale",
        ]
        assert params["prompt"].required
        assert params["prompt"].kind == "longtext"
        assert params["aspect_ratio"].default == "16:9"
        assert params["aspect_ratio"].enum == ("16:9", "9:16", "4:3", "3:4", "1:1")
        assert params["upscale"].default is False
        assert all(p.description for p in generate_image_tool.params)

    def test_description_disambiguates_from_the_sibling_tools(self) -> None:
        assert "edit_image" in generate_image_tool.description
        assert "gemini" not in generate_image_tool.description.lower()

    def test_description_states_pure_text_to_image(self) -> None:
        assert "brand-new image" in generate_image_tool.description
        assert "text prompt" in generate_image_tool.description
        assert "reference images" in generate_image_tool.description


class TestValidation:
    """Input checks happen before any run is created."""

    @pytest.mark.parametrize("prompt", ["", "   "])
    async def test_empty_prompt_rejected_without_run(self, prompt: str) -> None:
        result = await run_text(generate_image_tool, prompt)

        assert result == "prompt must be a non-empty string."


class TestPipeline:
    """The tool always runs the Qwen-Image-2.1 text to image pipeline."""

    async def test_runs_t2i(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, generate_image_tool, _CLIENT_PATH, "a cat")

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["30"]["inputs"]["filename_prefix"] == "media_gen_t2i"
        assert prompt["1"]["inputs"]["unet_name"] == "qwen_image_2.1_int8_convrot.safetensors"
        assert prompt["5"]["class_type"] == "TextEncodeQwenImage21"
        assert "LoadImage" not in {node["class_type"] for node in prompt.values()}

    async def test_aspect_ratio_applies_to_t2i(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(
            mock_client_cls, generate_image_tool, _CLIENT_PATH, "a cat",
            aspect_ratio="4:3",
        )

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["6"]["inputs"] == {"width": _EXP_T2I_4_3_W, "height": _EXP_T2I_4_3_H, "batch_size": 1}
        assert prompt["26"]["inputs"]["width"] == _EXP_T2I_4_3_W
        assert prompt["26"]["inputs"]["height"] == _EXP_T2I_4_3_H


class TestUpscaleAndSuccess:
    """The upscale flag, the saved file, and the cache key."""

    async def test_upscale_flag_reaches_the_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, generate_image_tool, _CLIENT_PATH, "a cat", upscale=True)

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["23"]["class_type"] == "UpscaleModelLoader"
        assert prompt["26"]["inputs"]["width"] == _EXP_T2I_16_9_2X_W
        assert prompt["26"]["inputs"]["height"] == _EXP_T2I_16_9_2X_H

    async def test_default_run_has_no_upscale_nodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, generate_image_tool, _CLIENT_PATH, "a dog")

        prompt = submitted_prompt(mock_client_cls)
        assert "23" not in prompt
        assert "24" not in prompt
        assert prompt["26"]["inputs"]["width"] == _EXP_T2I_16_9_W
        assert prompt["26"]["inputs"]["height"] == _EXP_T2I_16_9_H

    async def test_success_saves_under_the_tool_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        result = await run_media_tool(mock_client_cls, generate_image_tool, _CLIENT_PATH, "a cat")

        assert_saved_image(tmp_path, "generate_image_*.png", result, _PNG_BYTES)

    async def test_cache_key_changes_with_upscale(self) -> None:
        base = await generate_image_tool.cache_key({"prompt": "a cat"})
        upscaled = await generate_image_tool.cache_key({"prompt": "a cat", "upscale": True})

        assert base != upscaled

    async def test_cache_key_carries_pipeline_discriminator(self) -> None:
        """The t2i key differs from the old recipe's key for the same params."""
        key = await generate_image_tool.cache_key(
            {"prompt": "a cat", "aspect_ratio": "16:9", "upscale": True}
        )
        old_payload = json.dumps(
            {"prompt": "a cat", "aspect_ratio": "16:9", "upscale": True},
            ensure_ascii=False,
            sort_keys=True,
        )

        assert key != hashlib.sha256(old_payload.encode("utf-8")).hexdigest()


class TestModelFilenames:
    """The loader nodes name the default model files, or the configured ones."""

    @pytest.mark.parametrize(("node", "key", "variable", "default"), _T2I_MODEL_INPUTS)
    @pytest.mark.parametrize("configured", [False, True])
    async def test_loader_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        node: str, key: str, variable: str, default: str, configured: bool,
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        if configured:
            monkeypatch.setenv(variable, _CONFIGURED_MODEL)
        else:
            monkeypatch.delenv(variable, raising=False)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, generate_image_tool, _CLIENT_PATH, "a cat", upscale=True)

        expected = _CONFIGURED_MODEL if configured else default
        assert submitted_prompt(mock_client_cls)[node]["inputs"][key] == expected


class TestMissingBackend:
    """An unset backend address fails the run and names the variable."""

    async def test_unset_backend_fails_the_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
        point_resources_at(tmp_path, monkeypatch)
        monkeypatch.delenv("COMFYUI_BACKEND_URL", raising=False)

        result = await run_text(generate_image_tool, "a cat")

        assert result == (
            "Tool 'generate_image' failed: COMFYUI_BACKEND_URL is not set: "
            "needed by generate_image, edit_image, generate_video"
        )
        assert (
            "[config] COMFYUI_BACKEND_URL is not set: needed by generate_image, edit_image, generate_video\n"
            in capsys.readouterr().err
        )


@pytest.mark.live
@pytest.mark.timeout(1200)
async def test_generate_image_live_upscale_smoke() -> None:
    """A live upscale run exercises the opt-in RealESRGAN path."""
    if "DAV_BASE_URL" not in os.environ:
        pytest.skip("DAV_BASE_URL is not set")
    if not comfyui_online():
        pytest.skip("the ComfyUI backend is not reachable")
    saved: Path | None = None
    try:
        prompt = f"an upscaled test square {uuid.uuid4().hex}"
        saved = await live_image_run(generate_image_tool, prompt, upscale=True)
    finally:
        if saved is not None:
            saved.unlink(missing_ok=True)
