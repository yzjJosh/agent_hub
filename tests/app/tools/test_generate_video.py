"""Tests for app.tools.generate_video."""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.lib.shared.paths import resources_dir
from app.lib.task.task_manager import AsyncTask
from app.resources.resources import dav_base_url
from app.tools.generate_video import generate_video_tool
from tests.app.conftest import (
    DAV_BASE_URL,
    E2E_PNG,
    comfyui_online,
    point_resources_at,
    publish_reference,
    retract_reference,
    run_media_tool,
    submitted_prompt,
    wire_media_client,
    write_reference_image,
)
from tests.app.lib.tool.conftest import await_task_result, run_text

_MP4_BYTES = b"\x00\x00\x00\x18ftypisomgenerated video bytes"
_MP4_FILENAME = "media_gen_t2v/out_00001_.mp4"
_CLIENT_PATH = "app.tools.generate_video.ComfyUIClient"

# The video prompt's model loader inputs, the variable that names each, and
# the default filename.
_VIDEO_LOADERS = [
    (("1", "unet_name"), "COMFYUI_VIDEO_UNET", "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"),
    (("2", "clip_name"), "COMFYUI_VIDEO_TEXT_ENCODER", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    (("3", "vae_name"), "COMFYUI_VIDEO_VAE", "ltx-2.5-video-vae-bf16.safetensors"),
    (("4", "vae_name"), "COMFYUI_VIDEO_AUDIO_VAE", "ltx-2.5-audio-vae-bf16.safetensors"),
    (("23", "model_name"), "COMFYUI_UPSCALE_MODEL", "RealESRGAN_x2plus.pth"),
]
_VIDEO_MODES = {
    "t2v": {},
    "i2v": {"first_frame": "upload/ref.png"},
    "f2lv": {"first_frame": "upload/ref.png", "last_frame": "upload/last.png"},
}


class TestGenerateVideoShape:
    """The tool declares the video surface and nothing else."""

    def test_tool_shape(self) -> None:
        assert generate_video_tool.name == "generate_video"
        assert generate_video_tool.is_slow is True
        assert generate_video_tool.staleness == timedelta(hours=24)
        assert generate_video_tool.result_render_type == "text"
        assert generate_video_tool.summary
        assert "\n" not in generate_video_tool.summary
        assert len(generate_video_tool.summary) <= 80

    def test_params(self) -> None:
        params = {p.name: p for p in generate_video_tool.params}
        assert [p.name for p in generate_video_tool.params] == [
            "prompt",
            "first_frame",
            "last_frame",
            "aspect_ratio",
            "upscale",
        ]
        assert params["prompt"].required
        assert params["prompt"].kind == "longtext"
        assert params["first_frame"].default is None
        assert params["first_frame"].kind == "resource"
        assert params["last_frame"].default is None
        assert params["last_frame"].kind == "resource"
        assert params["aspect_ratio"].default == "16:9"
        assert params["aspect_ratio"].enum == ("16:9", "9:16", "4:3", "3:4", "1:1")
        assert params["upscale"].default is False
        assert all(p.description for p in generate_video_tool.params)

    def test_description_disambiguates_from_the_sibling_tools(self) -> None:
        assert "generate_image" in generate_video_tool.description
        assert "gemini" not in generate_video_tool.description.lower()


class TestValidation:
    """Input checks happen before any run is created."""

    @pytest.mark.parametrize("prompt", ["", "   "])
    async def test_empty_prompt_rejected_without_run(self, prompt: str) -> None:
        result = await run_text(generate_video_tool, prompt)

        assert result == "prompt must be a non-empty string."

    async def test_last_frame_without_first_frame_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        result = await run_media_tool(
            mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", last_frame="upload/ref.png"
        )

        assert result == "Tool 'generate_video' failed: last_frame requires first_frame."
        assert mock_client_cls.return_value.generate.await_count == 0

    async def test_missing_frame_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        result = await run_media_tool(
            mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", first_frame="upload/missing.png"
        )

        assert "Resource not found: upload/missing.png" in result
        assert mock_client_cls.return_value.generate.await_count == 0


class TestPipelineSelection:
    """The pipeline follows the given frames."""

    async def test_no_frames_runs_t2v(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        await run_media_tool(mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave")

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["30"]["inputs"]["filename_prefix"] == "media_gen_t2v"
        assert "LTXVAddGuide" not in {node["class_type"] for node in prompt.values()}
        assert "LoadImage" not in {node["class_type"] for node in prompt.values()}

    async def test_first_frame_runs_i2v(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        await run_media_tool(mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", first_frame="upload/ref.png")

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["30"]["inputs"]["filename_prefix"] == "media_gen_i2v"
        assert prompt["20"]["inputs"]["image"] == f"{DAV_BASE_URL}/dav/upload/ref.png"

    async def test_both_frames_run_f2lv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        (tmp_path / "upload" / "last.png").write_bytes(b"last frame")
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        await run_media_tool(
            mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", first_frame="upload/ref.png",
            last_frame="upload/last.png",
        )

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["30"]["inputs"]["filename_prefix"] == "media_gen_f2lv"
        assert prompt["40"]["inputs"]["image"] == f"{DAV_BASE_URL}/dav/upload/ref.png"
        assert prompt["41"]["inputs"]["image"] == f"{DAV_BASE_URL}/dav/upload/last.png"
        assert prompt["44"]["inputs"]["frame_idx"] == 0
        assert prompt["45"]["inputs"]["frame_idx"] == -1


class TestUpscaleAndCrop:
    """The upscale flag and the exact crop target reach the prompt."""

    async def test_default_run_is_base_resolution_without_upscale(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        await run_media_tool(mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", aspect_ratio="4:3")

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["10"]["inputs"] == {"width": 736, "height": 544, "length": 241, "batch_size": 1}
        assert prompt["26"]["inputs"]["width"] == 720
        assert prompt["26"]["inputs"]["height"] == 540
        assert "23" not in prompt
        assert "24" not in prompt

    async def test_upscale_run_doubles_and_crops(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        await run_media_tool(mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", aspect_ratio="4:3", upscale=True)

        prompt = submitted_prompt(mock_client_cls)
        assert prompt["26"]["inputs"]["width"] == 1448
        assert prompt["26"]["inputs"]["height"] == 1086
        assert prompt["23"]["class_type"] == "UpscaleModelLoader"


class TestModelFilenames:
    """Every video mode loads the default model files, or the configured ones."""

    @pytest.mark.parametrize("mode", list(_VIDEO_MODES))
    @pytest.mark.parametrize(("loader_input", "variable", "default"), _VIDEO_LOADERS)
    @pytest.mark.parametrize("configured", [False, True])
    async def test_loader_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        mode: str, loader_input: tuple[str, str], variable: str, default: str, configured: bool,
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        (tmp_path / "upload" / "last.png").write_bytes(b"last frame")
        if configured:
            monkeypatch.setenv(variable, f"{variable}.bin")
        else:
            monkeypatch.delenv(variable, raising=False)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        await run_media_tool(
            mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave", upscale=True, **_VIDEO_MODES[mode],
        )

        prompt = submitted_prompt(mock_client_cls)
        node, key = loader_input
        assert prompt["30"]["inputs"]["filename_prefix"] == f"media_gen_{mode}"
        assert prompt[node]["inputs"][key] == (f"{variable}.bin" if configured else default)


class TestSuccessAndCache:
    """The saved file, the result text, and the cache key."""

    async def test_success_saves_under_the_tool_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        mock_client_cls = MagicMock()
        wire_media_client(mock_client_cls, _MP4_BYTES, _MP4_FILENAME)

        result = await run_media_tool(mock_client_cls, generate_video_tool, _CLIENT_PATH, "a wave")

        saved = list((tmp_path / "media_gen").glob("generate_video_*.mp4"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == _MP4_BYTES
        resource_path = f"media_gen/{saved[0].name}"
        assert result.startswith(f"Video generated and saved to resource path: {resource_path}")
        assert f"{DAV_BASE_URL}/dav/{resource_path}" in result

    async def test_cache_key_changes_with_upscale_and_frames(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)

        base = await generate_video_tool.cache_key({"prompt": "a wave"})
        upscaled = await generate_video_tool.cache_key({"prompt": "a wave", "upscale": True})
        framed = await generate_video_tool.cache_key({"prompt": "a wave", "first_frame": "upload/ref.png"})

        assert base != upscaled
        assert base != framed

    async def test_cache_key_fingerprints_frame_contents(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        write_reference_image(tmp_path)
        (tmp_path / "upload" / "other.png").write_bytes(b"different frame")

        first = await generate_video_tool.cache_key({"prompt": "a wave", "first_frame": "upload/ref.png"})
        (tmp_path / "upload" / "ref.png").write_bytes(b"changed frame")
        changed = await generate_video_tool.cache_key({"prompt": "a wave", "first_frame": "upload/ref.png"})

        assert first != changed


_LIVE_GENERATION_WAIT_S = 900.0


@pytest.mark.live
@pytest.mark.timeout(1200)
async def test_generate_video_live_first_last_frame_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live first and last frame run exercises the f2lv guide chain end to end."""
    if "DAV_BASE_URL" not in os.environ:
        pytest.skip("DAV_BASE_URL is not set, the backend host cannot fetch the frames")
    if not comfyui_online():
        pytest.skip("the ComfyUI backend is not reachable")
    base_url = dav_base_url()
    first_name = f"generate_video_live_first_{uuid.uuid4().hex}.png"
    last_name = f"generate_video_live_last_{uuid.uuid4().hex}.png"
    first = f"upload/{first_name}"
    last = f"upload/{last_name}"
    saved: Path | None = None
    try:
        await publish_reference(base_url, first_name, E2E_PNG)
        await publish_reference(base_url, last_name, E2E_PNG)
        monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
        upload = tmp_path / "upload"
        upload.mkdir(parents=True)
        (upload / first_name).write_bytes(E2E_PNG)
        (upload / last_name).write_bytes(E2E_PNG)

        prompt = f"a slow pan across a test square {uuid.uuid4().hex}"
        task = await generate_video_tool.run({"prompt": prompt, "first_frame": first, "last_frame": last})
        assert isinstance(task, AsyncTask)
        result = await await_task_result(str(task.id), timeout_s=_LIVE_GENERATION_WAIT_S)

        assert result.startswith("Video generated and saved to resource path:")
        resource_path = result.split("resource path: ", 1)[1].split("\n", 1)[0]
        saved = Path(resources_dir()) / resource_path
        assert saved.exists()
    finally:
        await retract_reference(base_url, first_name)
        await retract_reference(base_url, last_name)
        if saved is not None:
            saved.unlink(missing_ok=True)
        (tmp_path / first).unlink(missing_ok=True)
        (tmp_path / last).unlink(missing_ok=True)
