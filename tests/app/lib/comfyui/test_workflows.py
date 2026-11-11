"""Tests for app.lib.comfyui.workflows."""

from __future__ import annotations

import inspect

import pytest

from app.lib.comfyui.workflows import (
    build_edit_prompt,
    build_f2lv_prompt,
    build_i2v_prompt,
    build_t2i_prompt,
    build_t2v_prompt,
)

_LTX_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
_QWEN_NEGATIVE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感"
)
_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"
_EXP_GUIDE_STRENGTH = 0.7
_EXP_IMG_COMPRESSION = 33

# Expected pixels for the base generation and crop targets.
_EXP_512 = 512
_EXP_540 = 540
_EXP_544 = 544
_EXP_576 = 576
_EXP_720 = 720
_EXP_736 = 736
_EXP_768 = 768
_EXP_1024 = 1024
_EXP_1086 = 1086
_EXP_1088 = 1088
_EXP_1152 = 1152
_EXP_1344 = 1344
_EXP_1448 = 1448
_EXP_1536 = 1536
_EXP_1792 = 1792
_EXP_2048 = 2048
_EXP_2304 = 2304
_EXP_2688 = 2688
_EXP_3584 = 3584
_EXP_4096 = 4096

# Crop target width and height per aspect ratio and upscale state. The target
# is the largest even exact-aspect-ratio fit inside the base box when upscale
# is False, or the 2x base box when upscale is True.
_VIDEO_CROP_TARGETS = {
    "16:9": ((_EXP_1024, _EXP_576), (_EXP_2048, _EXP_1152)),
    "9:16": ((_EXP_576, _EXP_1024), (_EXP_1152, _EXP_2048)),
    "4:3": ((_EXP_720, _EXP_540), (_EXP_1448, _EXP_1086)),
    "3:4": ((_EXP_540, _EXP_720), (_EXP_1086, _EXP_1448)),
    "1:1": ((_EXP_544, _EXP_544), (_EXP_1088, _EXP_1088)),
}

# Base generation width and height per aspect ratio.
_VIDEO_BASE = {
    "16:9": (_EXP_1024, _EXP_576),
    "9:16": (_EXP_576, _EXP_1024),
    "4:3": (_EXP_736, _EXP_544),
    "3:4": (_EXP_544, _EXP_736),
    "1:1": (_EXP_544, _EXP_544),
}
_T2I_BASE = {
    "16:9": (_EXP_2048, _EXP_1152),
    "9:16": (_EXP_1152, _EXP_2048),
    "4:3": (_EXP_1792, _EXP_1344),
    "3:4": (_EXP_1344, _EXP_1792),
    "1:1": (_EXP_2048, _EXP_2048),
}
_T2I_CROP_TARGETS = {
    "16:9": ((_EXP_2048, _EXP_1152), (_EXP_4096, _EXP_2304)),
    "9:16": ((_EXP_1152, _EXP_2048), (_EXP_2304, _EXP_4096)),
    "4:3": ((_EXP_1792, _EXP_1344), (_EXP_3584, _EXP_2688)),
    "3:4": ((_EXP_1344, _EXP_1792), (_EXP_2688, _EXP_3584)),
    "1:1": ((_EXP_2048, _EXP_2048), (_EXP_4096, _EXP_4096)),
}


def test_save_node_prefixes() -> None:
    assert build_t2v_prompt("p", 1)["30"]["inputs"]["filename_prefix"] == "media_gen_t2v"
    assert build_i2v_prompt("p", "u", 1)["30"]["inputs"]["filename_prefix"] == "media_gen_i2v"
    assert build_t2i_prompt("p", 1)["30"]["inputs"]["filename_prefix"] == "media_gen_t2i"
    assert build_edit_prompt("p", "u", 1, image_dims=(_EXP_2048, _EXP_2048))["30"]["inputs"]["filename_prefix"] == "media_gen_edit"


def _class_types(api: dict) -> set[str]:
    """The set of node classes in a built prompt."""
    return {node["class_type"] for node in api.values()}


def _prompt_texts(api: dict) -> list[str]:
    """The texts of every text-encoding node in a built prompt."""
    texts = []
    for node in api.values():
        if node["class_type"] == "CLIPTextEncode":
            texts.append(node["inputs"]["text"])
        elif node["class_type"] == "TextEncodeQwenImage21":
            texts.append(node["inputs"]["prompt"])
            texts.append(node["inputs"]["negative_prompt"])
    return texts


def _assert_crop_tail(api: dict, source_node: str, target_w: int, target_h: int) -> None:
    """The final ImageScale, wired from the decode (or upscale) node."""
    assert api["26"]["class_type"] == "ImageScale"
    assert api["26"]["inputs"]["image"] == [source_node, 0]
    assert api["26"]["inputs"]["upscale_method"] == "nearest-exact"
    assert api["26"]["inputs"]["width"] == target_w
    assert api["26"]["inputs"]["height"] == target_h
    assert api["26"]["inputs"]["crop"] == "center"


def _assert_upscale_pair(api: dict, source_node: str) -> None:
    """The 2x Real-ESRGAN pair wired from a decode node."""
    assert api["23"]["class_type"] == "UpscaleModelLoader"
    assert api["23"]["inputs"]["model_name"] == _UPSCALE_MODEL
    assert api["24"]["class_type"] == "ImageUpscaleWithModel"
    assert api["24"]["inputs"]["upscale_model"] == ["23", 0]
    assert api["24"]["inputs"]["image"] == [source_node, 0]


def _assert_crop_state(api: dict, target_w: int, target_h: int, upscale: bool) -> None:
    """The upscale pair and crop node in the state the flag selects."""
    if upscale:
        _assert_upscale_pair(api, "18" if "18" in api else "22")
        _assert_crop_tail(api, "24", target_w, target_h)
    else:
        assert "23" not in api
        assert "24" not in api
        decode = "18" if "18" in api else "22"
        _assert_crop_tail(api, decode, target_w, target_h)


class TestBuildT2vPrompt:
    """The LTX-2.5 text to video prompt."""

    def test_fills_prompt_and_seed(self) -> None:
        api = build_t2v_prompt("a lighthouse", 7)

        assert api["5"]["inputs"]["text"] == "a lighthouse"
        assert api["5"]["class_type"] == "CLIPTextEncode"
        assert api["13"]["inputs"]["noise_seed"] == 7
        assert api["13"]["class_type"] == "RandomNoise"

    def test_keeps_the_frozen_wiring_and_values(self) -> None:
        api = build_t2v_prompt("a lighthouse", 7)

        assert _class_types(api) == {
            "UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
            "LTXVConditioning", "LTXVDualCFGGuider", "EmptyLTXVLatentVideo",
            "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "RandomNoise",
            "KSamplerSelect", "ManualSigmas", "SamplerCustomAdvanced",
            "LTXVSeparateAVLatent", "VAEDecodeTiled", "LTXVAudioVAEDecode",
            "ImageScale", "CreateVideo", "SaveVideo",
        }
        assert _LTX_NEGATIVE in _prompt_texts(api)
        assert api["10"]["inputs"] == {"width": _EXP_1024, "height": _EXP_576, "length": 241, "batch_size": 1}
        assert api["11"]["inputs"]["frames_number"] == 241
        assert api["15"]["inputs"]["sigmas"] == _SIGMAS
        assert api["14"]["inputs"]["sampler_name"] == "euler_ancestral"
        assert api["16"]["inputs"]["latent_image"] == ["12", 0]
        assert api["25"]["inputs"]["fps"] == 24
        assert api["25"]["inputs"]["images"] == ["26", 0]
        assert api["25"]["inputs"]["audio"] == ["19", 0]
        assert api["30"]["inputs"]["filename_prefix"] == "media_gen_t2v"
        assert api["30"]["inputs"]["format"] == "auto"

    @pytest.mark.parametrize("aspect_ratio", list(_VIDEO_CROP_TARGETS))
    def test_sets_base_dimensions_for_each_ratio(self, aspect_ratio: str) -> None:
        base_w, base_h = _VIDEO_BASE[aspect_ratio]
        api = build_t2v_prompt("a lighthouse", 7, aspect_ratio)

        assert api["10"]["inputs"]["width"] == base_w
        assert api["10"]["inputs"]["height"] == base_h
        assert api["10"]["inputs"]["length"] == 241
        assert api["25"]["inputs"]["images"] == ["26", 0]

    @pytest.mark.parametrize("aspect_ratio", list(_VIDEO_CROP_TARGETS))
    @pytest.mark.parametrize("upscale", (False, True))
    def test_crop_target_follows_ratio_and_upscale(self, aspect_ratio: str, upscale: bool) -> None:
        base_w, base_h = _VIDEO_BASE[aspect_ratio]
        crop_w, crop_h = _VIDEO_CROP_TARGETS[aspect_ratio][1 if upscale else 0]
        api = build_t2v_prompt("a lighthouse", 7, aspect_ratio, upscale=upscale)

        assert api["10"]["inputs"]["width"] == base_w
        assert api["10"]["inputs"]["height"] == base_h
        _assert_crop_state(api, crop_w, crop_h, upscale)

    @pytest.mark.parametrize("upscale", (False, True))
    def test_upscale_pair_present_only_when_flag_set(self, upscale: bool) -> None:
        api = build_t2v_prompt("a lighthouse", 7, upscale=upscale)

        classes = {
            "UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
            "LTXVConditioning", "LTXVDualCFGGuider", "EmptyLTXVLatentVideo",
            "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "RandomNoise",
            "KSamplerSelect", "ManualSigmas", "SamplerCustomAdvanced",
            "LTXVSeparateAVLatent", "VAEDecodeTiled", "LTXVAudioVAEDecode",
            "ImageScale", "CreateVideo", "SaveVideo",
        }
        if upscale:
            classes |= {"UpscaleModelLoader", "ImageUpscaleWithModel"}
        assert _class_types(api) == classes
        assert ("23" in api) is upscale
        assert ("24" in api) is upscale

    def test_default_aspect_ratio_is_16_to_9(self) -> None:
        default = build_t2v_prompt("a lighthouse", 7)
        explicit = build_t2v_prompt("a lighthouse", 7, "16:9")

        assert default["10"]["inputs"] == explicit["10"]["inputs"]
        assert default["26"]["inputs"] == explicit["26"]["inputs"]

    def test_save_prefix_has_no_slash(self) -> None:
        api = build_t2v_prompt("a lighthouse", 7)

        assert "/" not in api["30"]["inputs"]["filename_prefix"]

    def test_has_no_image_node(self) -> None:
        assert "LoadImage" not in _class_types(build_t2v_prompt("a lighthouse", 7))

    def test_does_not_mutate_across_calls(self) -> None:
        first = build_t2v_prompt("one", 1)
        second = build_t2v_prompt("two", 2)

        assert first["5"]["inputs"]["text"] == "one"
        assert first["13"]["inputs"]["noise_seed"] == 1
        assert second["5"]["inputs"]["text"] == "two"
        assert second["13"]["inputs"]["noise_seed"] == 2


class TestBuildI2vPrompt:
    """The LTX-2.5 image to video prompt."""

    def test_fills_prompt_image_and_seed(self) -> None:
        api = build_i2v_prompt("waves roll in", "ref.png", 9)

        assert api["5"]["inputs"]["text"] == "waves roll in"
        assert api["20"]["class_type"] == "LoadImage"
        assert api["20"]["inputs"]["image"] == "ref.png"
        assert api["21"]["inputs"]["image"] == ["20", 0]
        assert api["21"]["inputs"]["img_compression"] == _EXP_IMG_COMPRESSION
        assert api["22"]["class_type"] == "LTXVImgToVideoInplace"
        assert api["22"]["inputs"]["strength"] == _EXP_GUIDE_STRENGTH
        assert api["22"]["inputs"]["bypass"] is False
        assert api["13"]["inputs"]["noise_seed"] == 9

    def test_reroutes_the_video_latent_through_the_image(self) -> None:
        api = build_i2v_prompt("waves roll in", "ref.png", 9)

        assert api["12"]["inputs"]["video_latent"] == ["22", 0]
        assert api["30"]["inputs"]["filename_prefix"] == "media_gen_i2v"

    @pytest.mark.parametrize("aspect_ratio", list(_VIDEO_CROP_TARGETS))
    @pytest.mark.parametrize("upscale", (False, True))
    def test_crop_target_follows_ratio_and_upscale(self, aspect_ratio: str, upscale: bool) -> None:
        base_w, base_h = _VIDEO_BASE[aspect_ratio]
        crop_w, crop_h = _VIDEO_CROP_TARGETS[aspect_ratio][1 if upscale else 0]
        api = build_i2v_prompt("waves roll in", "ref.png", 9, aspect_ratio, upscale=upscale)

        assert api["10"]["inputs"]["width"] == base_w
        assert api["10"]["inputs"]["height"] == base_h
        _assert_crop_state(api, crop_w, crop_h, upscale)
        assert api["25"]["inputs"]["images"] == ["26", 0]
        assert api["25"]["inputs"]["audio"] == ["19", 0]
        assert "/" not in api["30"]["inputs"]["filename_prefix"]

    def test_does_not_mutate_the_t2v_template(self) -> None:
        build_i2v_prompt("waves", "ref.png", 1)

        plain = build_t2v_prompt("plain", 2)
        assert "20" not in plain
        assert plain["12"]["inputs"]["video_latent"] == ["10", 0]
        assert plain["30"]["inputs"]["filename_prefix"] == "media_gen_t2v"


class TestBuildF2lvPrompt:
    """The LTX-2.5 first and last frame to video prompt."""

    def test_guide_chain_wiring(self) -> None:
        api = build_f2lv_prompt("a wave", "http://dav.test/dav/upload/first.png", "http://dav.test/dav/upload/last.png", 7)

        assert api["40"]["class_type"] == "LoadImage"
        assert api["40"]["inputs"]["image"] == "http://dav.test/dav/upload/first.png"
        assert api["41"]["class_type"] == "LoadImage"
        assert api["41"]["inputs"]["image"] == "http://dav.test/dav/upload/last.png"
        assert api["42"]["class_type"] == "LTXVPreprocess"
        assert api["42"]["inputs"] == {"image": ["40", 0], "img_compression": _EXP_IMG_COMPRESSION}
        assert api["43"]["class_type"] == "LTXVPreprocess"
        assert api["43"]["inputs"] == {"image": ["41", 0], "img_compression": _EXP_IMG_COMPRESSION}
        assert api["44"]["class_type"] == "LTXVAddGuide"
        assert api["44"]["inputs"] == {
            "positive": ["7", 0], "negative": ["7", 1], "vae": ["3", 0],
            "latent": ["10", 0], "image": ["42", 0], "frame_idx": 0,
            "strength": _EXP_GUIDE_STRENGTH,
        }
        assert api["45"]["class_type"] == "LTXVAddGuide"
        assert api["45"]["inputs"] == {
            "positive": ["44", 0], "negative": ["44", 1], "vae": ["3", 0],
            "latent": ["44", 2], "image": ["43", 0], "frame_idx": -1,
            "strength": _EXP_GUIDE_STRENGTH,
        }

    def test_last_guide_feeds_guider_and_extended_latent(self) -> None:
        api = build_f2lv_prompt("a wave", "u1", "u2", 7)

        assert api["8"]["inputs"]["positive"] == ["45", 0]
        assert api["8"]["inputs"]["negative"] == ["45", 1]
        assert api["12"]["inputs"]["video_latent"] == ["45", 2]
        assert api["10"]["class_type"] == "EmptyLTXVLatentVideo"

    def test_crop_guides_sit_on_the_decode_path(self) -> None:
        api = build_f2lv_prompt("a wave", "u1", "u2", 7)

        assert api["46"]["class_type"] == "LTXVCropGuides"
        assert api["46"]["inputs"] == {"positive": ["45", 0], "negative": ["45", 1], "latent": ["17", 0]}
        assert api["18"]["inputs"]["samples"] == ["46", 2]
        assert api["19"]["inputs"]["samples"] == ["17", 1]
        assert "LTXVImgToVideoInplace" not in _class_types(api)

    @pytest.mark.parametrize("ratio", ["16:9", "9:16", "4:3", "3:4", "1:1"])
    def test_base_and_crop_follow_the_ratio(self, ratio: str) -> None:
        api = build_f2lv_prompt("a wave", "u1", "u2", 7, aspect_ratio=ratio)
        crop_w, crop_h = _VIDEO_CROP_TARGETS[ratio][0]
        base_w, base_h = _VIDEO_BASE[ratio]

        assert api["10"]["inputs"]["width"] == base_w
        assert api["10"]["inputs"]["height"] == base_h
        assert api["26"]["inputs"]["width"] == crop_w
        assert api["26"]["inputs"]["height"] == crop_h

    def test_upscale_flag_and_save_prefix(self) -> None:
        api = build_f2lv_prompt("a wave", "u1", "u2", 7, upscale=True)
        assert api["30"]["inputs"]["filename_prefix"] == "media_gen_f2lv"
        _assert_crop_state(api, _EXP_2048, _EXP_1152, True)

    def test_default_has_no_upscale_pair(self) -> None:
        api = build_f2lv_prompt("a wave", "u1", "u2", 7)
        assert "23" not in api
        assert "24" not in api


class TestBuildT2iPrompt:
    """The Qwen-Image-2.1 text to image prompt."""

    def test_fills_prompt_and_seed(self) -> None:
        api = build_t2i_prompt("a fisherman", 42)

        assert api["5"]["class_type"] == "TextEncodeQwenImage21"
        assert api["5"]["inputs"]["prompt"] == "a fisherman"
        assert api["7"]["class_type"] == "KSampler"
        assert api["7"]["inputs"]["seed"] == 42

    def test_keeps_the_frozen_wiring_and_values(self) -> None:
        api = build_t2i_prompt("a fisherman", 42)

        assert _class_types(api) == {
            "UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21",
            "EmptyLatentImage", "KSampler", "VAEDecode", "SplitImageWithAlpha",
            "ImageScale", "SaveImage",
        }
        assert any("AI感" in t for t in _prompt_texts(api))
        assert api["1"]["inputs"] == {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}
        assert api["3"]["inputs"] == {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}
        assert api["4"]["inputs"] == {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}
        assert api["5"]["inputs"]["clip"] == ["3", 0]
        assert api["5"]["inputs"]["negative_prompt"] == _QWEN_NEGATIVE
        assert api["5"]["inputs"]["resolution"] == _EXP_2048
        assert api["6"]["inputs"] == {"width": _EXP_2048, "height": _EXP_1152, "batch_size": 1}
        assert api["7"]["inputs"]["model"] == ["1", 0]
        assert api["7"]["inputs"]["steps"] == 25
        assert api["7"]["inputs"]["cfg"] == 1.0
        assert api["7"]["inputs"]["sampler_name"] == "euler"
        assert api["7"]["inputs"]["scheduler"] == "simple"
        assert api["7"]["inputs"]["positive"] == ["5", 0]
        assert api["7"]["inputs"]["negative"] == ["5", 1]
        assert api["7"]["inputs"]["latent_image"] == ["6", 0]
        assert api["7"]["inputs"]["denoise"] == 1.0
        assert api["8"]["inputs"] == {"samples": ["7", 0], "vae": ["4", 0]}
        assert api["22"]["inputs"] == {"image": ["8", 0]}
        assert api["30"]["inputs"]["filename_prefix"] == "media_gen_t2i"

    def test_old_recipe_is_gone(self) -> None:
        api = build_t2i_prompt("a fisherman", 42)
        leftover = _class_types(api) & {
            "UnetLoaderGGUF", "LoraLoaderModelOnly", "ModelSamplingAuraFlow",
            "CFGNorm", "CLIPTextEncode", "EmptySD3LatentImage",
        }
        assert not leftover, sorted(leftover)

    @pytest.mark.parametrize("aspect_ratio", list(_T2I_CROP_TARGETS))
    def test_sets_base_dimensions_for_each_ratio(self, aspect_ratio: str) -> None:
        base_w, base_h = _T2I_BASE[aspect_ratio]
        api = build_t2i_prompt("a fisherman", 42, aspect_ratio)

        assert api["6"]["inputs"]["width"] == base_w
        assert api["6"]["inputs"]["height"] == base_h
        assert api["30"]["inputs"]["images"] == ["26", 0]

    @pytest.mark.parametrize("aspect_ratio", list(_T2I_CROP_TARGETS))
    def test_resolution_matches_the_long_base_side(self, aspect_ratio: str) -> None:
        base_w, base_h = _T2I_BASE[aspect_ratio]
        api = build_t2i_prompt("a fisherman", 42, aspect_ratio)

        assert api["5"]["inputs"]["resolution"] == max(base_w, base_h)

    @pytest.mark.parametrize("aspect_ratio", list(_T2I_CROP_TARGETS))
    @pytest.mark.parametrize("upscale", (False, True))
    def test_crop_target_follows_ratio_and_upscale(self, aspect_ratio: str, upscale: bool) -> None:
        base_w, base_h = _T2I_BASE[aspect_ratio]
        crop_w, crop_h = _T2I_CROP_TARGETS[aspect_ratio][1 if upscale else 0]
        api = build_t2i_prompt("a fisherman", 42, aspect_ratio, upscale=upscale)

        assert api["6"]["inputs"]["width"] == base_w
        assert api["6"]["inputs"]["height"] == base_h
        _assert_crop_state(api, crop_w, crop_h, upscale)
        assert api["30"]["inputs"]["images"] == ["26", 0]

    def test_save_prefix_has_no_slash(self) -> None:
        api = build_t2i_prompt("a fisherman", 42)

        assert "/" not in api["30"]["inputs"]["filename_prefix"]

    def test_has_no_image_node(self) -> None:
        assert "LoadImage" not in _class_types(build_t2i_prompt("a fisherman", 42))


class TestBuildEditPrompt:
    """The Qwen-Image-2.1 edit prompt."""

    def test_fills_instruction_image_and_seed(self) -> None:
        api = build_edit_prompt("repaint it white", "source.png", 5, image_dims=(_EXP_1536, _EXP_2048))

        assert api["8"]["class_type"] == "TextEncodeQwenImage21"
        assert api["8"]["inputs"]["prompt"] == "repaint it white"
        assert api["6"]["class_type"] == "LoadImage"
        assert api["6"]["inputs"]["image"] == "source.png"
        assert api["33"]["class_type"] == "ImageScale"
        assert api["33"]["inputs"]["image"] == ["6", 0]
        assert api["33"]["inputs"]["width"] == _EXP_1536
        assert api["33"]["inputs"]["height"] == _EXP_2048
        assert api["8"]["inputs"]["images.image_1"] == ["33", 0]
        assert api["7"]["inputs"]["seed"] == 5

    def test_keeps_the_frozen_wiring_and_values(self) -> None:
        api = build_edit_prompt("repaint it white", "source.png", 5, image_dims=(_EXP_1536, _EXP_2048))

        assert _class_types(api) == {
            "UNETLoader", "QwenImage21Cache", "CLIPLoader", "VAELoader",
            "LoadImage", "ImageScale", "TextEncodeQwenImage21", "KSampler", "VAEDecode",
            "SplitImageWithAlpha", "SaveImage",
        }
        assert api["1"]["inputs"] == {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}
        assert api["2"]["inputs"] == {"model": ["1", 0], "device": "auto", "dtype": "default"}
        assert api["3"]["inputs"] == {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}
        assert api["4"]["inputs"] == {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}
        assert api["33"]["inputs"] == {
            "image": ["6", 0], "upscale_method": "lanczos",
            "width": _EXP_1536, "height": _EXP_2048, "crop": "disabled",
        }
        assert api["8"]["inputs"]["prompt"] == "repaint it white"
        assert api["8"]["inputs"]["negative_prompt"] == _QWEN_NEGATIVE
        assert api["8"]["inputs"]["vae"] == ["4", 0]
        assert api["8"]["inputs"]["resolution"] == 0
        assert api["7"]["inputs"]["model"] == ["2", 0]
        assert api["7"]["inputs"]["steps"] == 25
        assert api["7"]["inputs"]["cfg"] == 1.0
        assert api["7"]["inputs"]["sampler_name"] == "euler"
        assert api["7"]["inputs"]["scheduler"] == "simple"
        assert api["7"]["inputs"]["positive"] == ["8", 0]
        assert api["7"]["inputs"]["negative"] == ["8", 1]
        assert api["7"]["inputs"]["latent_image"] == ["8", 2]
        assert api["7"]["inputs"]["denoise"] == 1.0
        assert api["9"]["inputs"] == {"samples": ["7", 0], "vae": ["4", 0]}
        assert api["22"]["inputs"] == {"image": ["9", 0]}
        assert api["30"]["inputs"]["filename_prefix"] == "media_gen_edit"

    def test_old_recipe_is_gone(self) -> None:
        api = build_edit_prompt("repaint it white", "source.png", 5, image_dims=(_EXP_1536, _EXP_2048))
        leftover = _class_types(api) & {
            "UnetLoaderGGUF", "ModelSamplingAuraFlow", "CFGNorm",
            "FluxKontextImageScale", "TextEncodeQwenImageEditPlus",
            "FluxKontextMultiReferenceLatentMethod", "VAEEncode", "LoraLoaderModelOnly",
        }
        assert not leftover, sorted(leftover)

    def test_decode_feeds_the_save_node(self) -> None:
        api = build_edit_prompt("repaint it white", "source.png", 5, image_dims=(_EXP_1536, _EXP_2048))

        assert api["22"]["class_type"] == "SplitImageWithAlpha"
        assert api["30"]["inputs"]["images"] == ["22", 0]
        assert "/" not in api["30"]["inputs"]["filename_prefix"]

    def test_no_upscale_nodes_at_all(self) -> None:
        api = build_edit_prompt("make it blue", "http://dav.test/dav/upload/ref.png", 7, image_dims=(_EXP_1024, _EXP_1024))

        assert "23" not in api
        assert "24" not in api
        assert "UpscaleModelLoader" not in _class_types(api)
        assert "ImageUpscaleWithModel" not in _class_types(api)
        assert api["30"]["inputs"]["images"] == ["22", 0]

    def test_takes_no_aspect_ratio(self) -> None:
        params = inspect.signature(build_edit_prompt).parameters

        assert "aspect_ratio" not in params

    def test_encoder_reads_only_the_scaled_images(self) -> None:
        api = build_edit_prompt(
            "repaint it white", "source.png", 5,
            image_dims=(_EXP_1536, _EXP_2048),
            extra_images=("a.png", "b.png"),
            extra_image_dims=((_EXP_1024, _EXP_1024), (_EXP_512, _EXP_768)),
        )

        image_inputs = {
            field: value
            for field, value in api["8"]["inputs"].items()
            if field.startswith("images.")
        }
        assert image_inputs == {
            "images.image_1": ["33", 0],
            "images.image_2": ["34", 0],
            "images.image_3": ["35", 0],
        }

    def test_one_extra_reference_wires_image2(self) -> None:
        api = build_edit_prompt(
            "repaint it white", "source.png", 5,
            image_dims=(_EXP_1536, _EXP_2048),
            extra_images=("style.png",),
            extra_image_dims=((_EXP_512, _EXP_512),),
        )

        assert api["31"]["class_type"] == "LoadImage"
        assert api["31"]["inputs"]["image"] == "style.png"
        assert api["34"]["class_type"] == "ImageScale"
        assert api["34"]["inputs"]["image"] == ["31", 0]
        assert api["34"]["inputs"]["width"] == _EXP_512
        assert api["34"]["inputs"]["height"] == _EXP_512
        assert api["8"]["inputs"]["images.image_2"] == ["34", 0]
        assert "images.image_3" not in api["8"]["inputs"]
        assert "32" not in api
        assert "35" not in api

    def test_two_extra_references_wire_image2_and_image3(self) -> None:
        api = build_edit_prompt(
            "repaint it white", "source.png", 5,
            image_dims=(_EXP_1536, _EXP_2048),
            extra_images=("a.png", "b.png"),
            extra_image_dims=((_EXP_1024, _EXP_1024), (_EXP_512, _EXP_768)),
        )

        assert api["31"]["inputs"]["image"] == "a.png"
        assert api["32"]["inputs"]["image"] == "b.png"
        assert api["34"]["inputs"]["image"] == ["31", 0]
        assert api["34"]["inputs"]["width"] == _EXP_1024
        assert api["35"]["inputs"]["image"] == ["32", 0]
        assert api["35"]["inputs"]["height"] == _EXP_768
        assert api["8"]["inputs"]["images.image_2"] == ["34", 0]
        assert api["8"]["inputs"]["images.image_3"] == ["35", 0]

    def test_no_extras_builds_the_same_prompt(self) -> None:
        assert build_edit_prompt("repaint it white", "source.png", 5, image_dims=(_EXP_1536, _EXP_2048)) == build_edit_prompt(
            "repaint it white", "source.png", 5, image_dims=(_EXP_1536, _EXP_2048), extra_images=()
        )

    def test_extras_loads_feed_only_their_scale_nodes(self) -> None:
        api = build_edit_prompt(
            "repaint it white", "source.png", 5,
            image_dims=(_EXP_1536, _EXP_2048),
            extra_images=("a.png", "b.png"),
            extra_image_dims=((_EXP_1024, _EXP_1024), (_EXP_512, _EXP_768)),
        )

        consumers = {
            (node_id, field)
            for node_id, node in api.items()
            for field, value in node["inputs"].items()
            if isinstance(value, list) and value[0] in ("31", "32")
        }
        assert consumers == {
            ("34", "image"), ("35", "image"),
        }

    def test_more_than_two_extras_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_edit_prompt(
                "repaint it white", "source.png", 5,
                image_dims=(_EXP_1536, _EXP_2048), extra_images=("a", "b", "c"),
            )
