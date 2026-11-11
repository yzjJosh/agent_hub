"""Tests for app.tools.edit_image."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.lib.comfyui.media import reference_fingerprints
from app.resources.resources import dav_base_url
from app.tools.edit_image import edit_image_tool
from tests.app.conftest import (
    DAV_BASE_URL,
    E2E_PNG,
    assert_saved_image,
    live_image_run,
    point_resources_at,
    publish_reference,
    require_live_media,
    retract_reference,
    run_media_tool,
    stage_reference_local,
    submitted_prompt,
    wire_media_client,
    write_reference_image,
    write_reference_png,
)
from tests.app.lib.tool.conftest import run_text

_PNG_BYTES = b"\x89PNG\r\n\x1aedited image bytes"
_PNG_FILENAME = "media_gen_edit/out_00001_.png"
_CLIENT_PATH = "app.tools.edit_image.ComfyUIClient"

# Test photo pixels and the fitted pixels the 2048 box gives them.
_UNIT_PX = 1024
_SQUARE_REF_PX = 2048
_OVERSIZED_W = 4096
_OVERSIZED_H = 3072
_FITTED_W = 2048
_FITTED_H = 1536

# Each loader of the edit prompt, with its variable, node, input and default filename.
_EDIT_LOADERS = [
    ("COMFYUI_IMAGE_UNET", "1", "unet_name", "qwen_image_2.1_int8_convrot.safetensors"),
    ("COMFYUI_IMAGE_CLIP", "3", "clip_name", "qwen3vl_8b_w4a8.safetensors"),
    ("COMFYUI_IMAGE_VAE", "4", "vae_name", "qwen_image_2.1_vae_bf16.safetensors"),
]


def _write_references(tmp_path: Path, count: int) -> list[str]:
    """Write count reference images into the upload folder and return their resource paths."""
    sizes = [(_SQUARE_REF_PX, _SQUARE_REF_PX), (_OVERSIZED_W, _OVERSIZED_H)]
    refs: list[str] = []
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        refs.append(write_reference_png(tmp_path, f"ref_{index}.png", width, height))
    return refs


def _assert_references_wired(prompt: dict, refs: list[str]) -> None:
    """Each reference loads into a LoadImage node its scale node feeds the encoder."""
    node_ids = ("31", "32")
    scale_ids = ("34", "35")
    fitted = {0: (_SQUARE_REF_PX, _SQUARE_REF_PX), 1: (_FITTED_W, _FITTED_H)}
    for index, reference in enumerate(refs):
        node_id = node_ids[index]
        scale_id = scale_ids[index]
        width, height = fitted[index]
        assert prompt[node_id]["class_type"] == "LoadImage"
        assert prompt[node_id]["inputs"]["image"] == f"{DAV_BASE_URL}/dav/{reference}"
        assert prompt[scale_id]["class_type"] == "ImageScale"
        assert prompt[scale_id]["inputs"]["image"] == [node_id, 0]
        assert prompt[scale_id]["inputs"]["width"] == width
        assert prompt[scale_id]["inputs"]["height"] == height
        assert prompt["8"]["inputs"][f"images.image_{index + 2}"] == [scale_id, 0]


class TestEditImageShape:
    """The tool takes an image, a prompt, and optional references."""

    def test_tool_shape(self) -> None:
        assert edit_image_tool.name == "edit_image"
        assert edit_image_tool.is_slow is True
        assert edit_image_tool.staleness == timedelta(hours=24)
        assert edit_image_tool.result_render_type == "text"
        assert edit_image_tool.summary
        assert "\n" not in edit_image_tool.summary
        assert len(edit_image_tool.summary) <= 80

    def test_params(self) -> None:
        params = {p.name: p for p in edit_image_tool.params}
        assert [p.name for p in edit_image_tool.params] == ["image", "prompt", "reference_images"]
        assert params["image"].required
        assert params["image"].kind == "resource"
        assert params["prompt"].required
        assert params["prompt"].kind == "longtext"
        assert params["reference_images"].default is None
        assert params["reference_images"].kind == "resource_list"
        assert all(p.description for p in edit_image_tool.params)

    def test_description_disambiguates_from_the_sibling_tools(self) -> None:
        assert "generate_image" in edit_image_tool.description
        assert "gemini" not in edit_image_tool.description.lower()


class TestEditRun:
    """The edit pipeline, the missing-image error, and the saved file."""

    @pytest.mark.parametrize("prompt", ["", "   "])
    async def test_empty_prompt_rejected_without_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)

        result = await run_text(edit_image_tool, "upload/ref.png", prompt)

        assert result == "prompt must be a non-empty string."

    async def test_missing_image_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        result = await run_media_tool(mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/missing.png", "make it blue")

        assert "Resource not found: upload/missing.png" in result
        assert mock_client_cls.return_value.generate.await_count == 0

    async def test_edit_prompt_scales_the_canvas_into_the_encoder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_png(tmp_path, "ref.png", _OVERSIZED_W, _OVERSIZED_H)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue")

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["30"]["inputs"]["filename_prefix"] == "media_gen_edit"
        assert prompt["6"]["inputs"]["image"] == f"{DAV_BASE_URL}/dav/upload/ref.png"
        assert prompt["33"]["class_type"] == "ImageScale"
        assert prompt["33"]["inputs"]["image"] == ["6", 0]
        assert prompt["33"]["inputs"]["width"] == _FITTED_W
        assert prompt["33"]["inputs"]["height"] == _FITTED_H
        assert prompt["8"]["class_type"] == "TextEncodeQwenImage21"
        assert prompt["8"]["inputs"]["images.image_1"] == ["33", 0]
        assert prompt["22"]["class_type"] == "SplitImageWithAlpha"
        assert "23" not in prompt
        assert "24" not in prompt
        assert prompt["30"]["inputs"]["images"] == ["22", 0]

    async def test_success_saves_under_the_tool_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_png(tmp_path, "ref.png", _UNIT_PX, _UNIT_PX)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        result = await run_media_tool(mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue")

        assert_saved_image(tmp_path, "edit_image_*.png", result, _PNG_BYTES)

    async def test_cache_key_fingerprints_the_image(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)

        first = await edit_image_tool.cache_key({"image": "upload/ref.png", "prompt": "make it blue"})
        (tmp_path / "upload" / "ref.png").write_bytes(b"changed image")
        changed = await edit_image_tool.cache_key({"image": "upload/ref.png", "prompt": "make it blue"})

        assert first != changed

    async def test_cache_key_carries_pipeline_discriminator(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The key differs from the old recipe's key for the same params."""
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)

        key = await edit_image_tool.cache_key({"image": "upload/ref.png", "prompt": "make it blue"})
        old_payload = json.dumps(
            {
                "prompt": "make it blue",
                "image": await reference_fingerprints(["upload/ref.png"]),
                "reference_images": await reference_fingerprints([]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        assert key != hashlib.sha256(old_payload.encode("utf-8")).hexdigest()


class TestModelFilenames:
    """The edit loaders name the default model files unless a variable names others."""

    @pytest.mark.parametrize(("variable", "node", "key", "default"), _EDIT_LOADERS)
    async def test_unset_variable_keeps_the_default_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, node: str, key: str, default: str,
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_png(tmp_path, "ref.png", _UNIT_PX, _UNIT_PX)
        monkeypatch.delenv(variable, raising=False)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue")

        assert submitted_prompt(mock_client_cls)[node]["inputs"][key] == default

    @pytest.mark.parametrize(("variable", "node", "key", "_default"), _EDIT_LOADERS)
    async def test_set_variable_names_the_configured_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, node: str, key: str, _default: str,
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_png(tmp_path, "ref.png", _UNIT_PX, _UNIT_PX)
        monkeypatch.setenv(variable, "edit_custom.safetensors")
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue")

        assert submitted_prompt(mock_client_cls)[node]["inputs"][key] == "edit_custom.safetensors"


class TestReferences:
    """The reference images guide the edit without taking over the canvas."""

    async def test_third_reference_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        refs = _write_references(tmp_path, 3)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        result = await run_media_tool(
            mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue",
            reference_images=refs,
        )

        assert "image editing takes at most 2 reference images, got 3" in result
        assert mock_client_cls.return_value.generate.await_count == 0

    async def test_missing_reference_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        result = await run_media_tool(
            mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue",
            reference_images=["upload/missing.png"],
        )

        assert "Resource not found: upload/missing.png" in result
        assert mock_client_cls.return_value.generate.await_count == 0

    @pytest.mark.parametrize("count", [1, 2])
    async def test_references_wire_the_spare_encoders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_png(tmp_path, "ref.png", _UNIT_PX, _UNIT_PX)
        refs = _write_references(tmp_path, count)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _PNG_BYTES, _PNG_FILENAME)

        await run_media_tool(
            mock_client_cls, edit_image_tool, _CLIENT_PATH, "upload/ref.png", "make it blue",
            reference_images=refs,
        )

        _assert_references_wired(submitted_prompt(mock_client_cls), refs)

    async def test_cache_key_fingerprints_reference_contents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        (tmp_path / "upload" / "ref_0.png").write_bytes(b"reference")

        first = await edit_image_tool.cache_key(
            {"image": "upload/ref.png", "prompt": "make it blue", "reference_images": ["upload/ref_0.png"]}
        )
        (tmp_path / "upload" / "ref_0.png").write_bytes(b"changed reference")
        changed = await edit_image_tool.cache_key(
            {"image": "upload/ref.png", "prompt": "make it blue", "reference_images": ["upload/ref_0.png"]}
        )

        assert first != changed



@pytest.mark.live
@pytest.mark.timeout(1200)
async def test_edit_image_live_reference_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live run with two references exercises the extra encoder slots end to end."""
    require_live_media()
    base_url = dav_base_url()
    image_name = f"edit_image_live_{uuid.uuid4().hex}.png"
    ref_a_name = f"edit_image_ref_a_{uuid.uuid4().hex}.png"
    ref_b_name = f"edit_image_ref_b_{uuid.uuid4().hex}.png"
    image = f"upload/{image_name}"
    ref_a = f"upload/{ref_a_name}"
    ref_b = f"upload/{ref_b_name}"
    saved: Path | None = None
    try:
        await publish_reference(base_url, image_name, E2E_PNG)
        await publish_reference(base_url, ref_a_name, E2E_PNG)
        await publish_reference(base_url, ref_b_name, E2E_PNG)
        stage_reference_local(tmp_path, monkeypatch, image, E2E_PNG)
        stage_reference_local(tmp_path, monkeypatch, ref_a, E2E_PNG)
        stage_reference_local(tmp_path, monkeypatch, ref_b, E2E_PNG)
        prompt = f"make the square blue {uuid.uuid4().hex}"
        saved = await live_image_run(
            edit_image_tool, image, prompt=prompt, reference_images=[ref_a, ref_b]
        )
    finally:
        await retract_reference(base_url, image_name)
        await retract_reference(base_url, ref_a_name)
        await retract_reference(base_url, ref_b_name)
        if saved is not None:
            saved.unlink(missing_ok=True)
        (tmp_path / image).unlink(missing_ok=True)
        (tmp_path / ref_a).unlink(missing_ok=True)
        (tmp_path / ref_b).unlink(missing_ok=True)
