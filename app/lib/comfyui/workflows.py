"""API-format prompt templates for the five generation pipelines."""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence

__all__ = [
    "build_t2i_prompt", "build_t2v_prompt", "build_i2v_prompt",
    "build_f2lv_prompt", "build_edit_prompt",
]

# Crop target pixels per video aspect ratio and upscale state. The target is
# the largest even exact-aspect-ratio fit inside the base box when upscale is
# False, or the 2x base box when upscale is True.
_VID_CROP_16_9_W = 1024
_VID_CROP_16_9_H = 576
_VID_CROP_16_9_2X_W = 2048
_VID_CROP_16_9_2X_H = 1152
_VID_CROP_4_3_W = 720
_VID_CROP_4_3_H = 540
_VID_CROP_4_3_2X_W = 1448
_VID_CROP_4_3_2X_H = 1086
_VID_CROP_1_1 = 544
_VID_CROP_1_1_2X = 1088

_VIDEO_CROP_TARGETS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "16:9": ((_VID_CROP_16_9_W, _VID_CROP_16_9_H), (_VID_CROP_16_9_2X_W, _VID_CROP_16_9_2X_H)),
    "9:16": ((_VID_CROP_16_9_H, _VID_CROP_16_9_W), (_VID_CROP_16_9_2X_H, _VID_CROP_16_9_2X_W)),
    "4:3": ((_VID_CROP_4_3_W, _VID_CROP_4_3_H), (_VID_CROP_4_3_2X_W, _VID_CROP_4_3_2X_H)),
    "3:4": ((_VID_CROP_4_3_H, _VID_CROP_4_3_W), (_VID_CROP_4_3_2X_H, _VID_CROP_4_3_2X_W)),
    "1:1": ((_VID_CROP_1_1, _VID_CROP_1_1), (_VID_CROP_1_1_2X, _VID_CROP_1_1_2X)),
}

# LTX-2.5 video base pixels. All values are multiples of 32.
_VID_BASE_16_9_W = 1024
_VID_BASE_16_9_H = 576
_VID_BASE_4_3_W = 736
_VID_BASE_4_3_H = 544
_VID_BASE_1_1 = 544

# Qwen-Image-2.1 text to image base pixels. All exact ratios, multiples of 32.
_T2I_BASE_16_9_W = 2048
_T2I_BASE_16_9_H = 1152
_T2I_BASE_4_3_W = 1792
_T2I_BASE_4_3_H = 1344
_T2I_BASE_1_1 = 2048

_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"
_GUIDE_STRENGTH = 0.7
_IMG_COMPRESSION = 33
_QWEN_UNET = "qwen_image_2.1_int8_convrot.safetensors"
_QWEN_CLIP = "qwen3vl_8b_w4a8.safetensors"
_QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
_LTX_UNET = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
_LTX_TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
_LTX_VAE = "ltx-2.5-video-vae-bf16.safetensors"
_LTX_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"

# The optional variables that name the model files on the ComfyUI backend.
# Unset or empty, each falls back to the filename above it.
_UPSCALE_MODEL_VAR = "COMFYUI_UPSCALE_MODEL"
_IMAGE_UNET_VAR = "COMFYUI_IMAGE_UNET"
_IMAGE_CLIP_VAR = "COMFYUI_IMAGE_CLIP"
_IMAGE_VAE_VAR = "COMFYUI_IMAGE_VAE"
_VIDEO_UNET_VAR = "COMFYUI_VIDEO_UNET"
_VIDEO_TEXT_ENCODER_VAR = "COMFYUI_VIDEO_TEXT_ENCODER"
_VIDEO_VAE_VAR = "COMFYUI_VIDEO_VAE"
_VIDEO_AUDIO_VAE_VAR = "COMFYUI_VIDEO_AUDIO_VAE"


_VIDEO_BASE_PIXELS: dict[str, tuple[int, int]] = {
    "16:9": (_VID_BASE_16_9_W, _VID_BASE_16_9_H),
    "9:16": (_VID_BASE_16_9_H, _VID_BASE_16_9_W),
    "4:3": (_VID_BASE_4_3_W, _VID_BASE_4_3_H),
    "3:4": (_VID_BASE_4_3_H, _VID_BASE_4_3_W),
    "1:1": (_VID_BASE_1_1, _VID_BASE_1_1),
}

_T2I_BASE_PIXELS: dict[str, tuple[int, int]] = {
    "16:9": (_T2I_BASE_16_9_W, _T2I_BASE_16_9_H),
    "9:16": (_T2I_BASE_16_9_H, _T2I_BASE_16_9_W),
    "4:3": (_T2I_BASE_4_3_W, _T2I_BASE_4_3_H),
    "3:4": (_T2I_BASE_4_3_H, _T2I_BASE_4_3_W),
    "1:1": (_T2I_BASE_1_1, _T2I_BASE_1_1),
}

# The t2i crop is the base itself when upscale is False, the 2x base box when
# upscale is True.
_T2I_CROP_TARGETS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    ratio: (base, (base[0] * 2, base[1] * 2)) for ratio, base in _T2I_BASE_PIXELS.items()
}


def _video_base(aspect_ratio: str) -> tuple[int, int]:
    """The base generation pixels for a video aspect ratio."""
    return _VIDEO_BASE_PIXELS[aspect_ratio]


def _t2i_base(aspect_ratio: str) -> tuple[int, int]:
    """The base generation pixels for a text to image aspect ratio."""
    return _T2I_BASE_PIXELS[aspect_ratio]


def _crop_target(table: dict[str, tuple[tuple[int, int], tuple[int, int]]], aspect_ratio: str, upscale: bool) -> tuple[int, int]:
    """The exact-aspect-ratio crop pixels for a ratio and upscale state."""
    return table[aspect_ratio][1 if upscale else 0]


def _set_crop_tail(api: dict, decode_node: str, target: tuple[int, int], upscale: bool) -> None:
    """Wire the optional upscale pair and the final center-crop for a template."""
    crop_w, crop_h = target
    api["26"]["inputs"]["width"] = crop_w
    api["26"]["inputs"]["height"] = crop_h
    if upscale:
        api["26"]["inputs"]["image"] = ["24", 0]
        api["24"]["inputs"]["image"] = [decode_node, 0]
        api["23"]["inputs"]["model_name"] = os.getenv(_UPSCALE_MODEL_VAR) or _UPSCALE_MODEL
    else:
        api.pop("23", None)
        api.pop("24", None)
        api["26"]["inputs"]["image"] = [decode_node, 0]


def _set_image_models(api: dict) -> None:
    """Name the configured Qwen-Image model files in a t2i or edit prompt."""
    api["1"]["inputs"]["unet_name"] = os.getenv(_IMAGE_UNET_VAR) or _QWEN_UNET
    api["3"]["inputs"]["clip_name"] = os.getenv(_IMAGE_CLIP_VAR) or _QWEN_CLIP
    api["4"]["inputs"]["vae_name"] = os.getenv(_IMAGE_VAE_VAR) or _QWEN_VAE


def _set_video_models(api: dict) -> None:
    """Name the configured LTX model files in a video prompt."""
    api["1"]["inputs"]["unet_name"] = os.getenv(_VIDEO_UNET_VAR) or _LTX_UNET
    api["2"]["inputs"]["clip_name"] = os.getenv(_VIDEO_TEXT_ENCODER_VAR) or _LTX_TEXT_ENCODER
    api["3"]["inputs"]["vae_name"] = os.getenv(_VIDEO_VAE_VAR) or _LTX_VAE
    api["4"]["inputs"]["vae_name"] = os.getenv(_VIDEO_AUDIO_VAE_VAR) or _LTX_AUDIO_VAE


_LTX_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
_EXTRA_IMAGE_NODE_IDS = ("31", "32")
_EXTRA_IMAGE_SCALE_NODE_IDS = ("34", "35")
_EDIT_SCALE_NODE_ID = "33"
_EDIT_SCALE_METHOD = "lanczos"
_T2I_NEGATIVE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感"
)
_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

_T2V = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": _LTX_UNET,
            "weight_dtype": "default",
        },
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": _LTX_TEXT_ENCODER,
            "type": "ltxv",
            "device": "default",
        },
    },
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": _LTX_VAE}},
    "4": {"class_type": "VAELoader", "inputs": {"vae_name": _LTX_AUDIO_VAE}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": _LTX_NEGATIVE, "clip": ["2", 0]}},
    "7": {
        "class_type": "LTXVConditioning",
        "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24},
    },
    "8": {
        "class_type": "LTXVDualCFGGuider",
        "inputs": {
            "model": ["1", 0],
            "positive": ["7", 0],
            "negative": ["7", 1],
            "video_cfg": 1.0,
            "audio_cfg": 1.0,
        },
    },
    "10": {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {"width": _VID_BASE_16_9_W, "height": _VID_BASE_16_9_H, "length": 241, "batch_size": 1},
    },
    "11": {
        "class_type": "LTXVEmptyLatentAudio",
        "inputs": {"frames_number": 241, "frame_rate": 24, "batch_size": 1, "audio_vae": ["4", 0]},
    },
    "12": {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {"video_latent": ["10", 0], "audio_latent": ["11", 0]},
    },
    "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
    "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
    "15": {"class_type": "ManualSigmas", "inputs": {"sigmas": _SIGMAS}},
    "16": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["13", 0],
            "guider": ["8", 0],
            "sampler": ["14", 0],
            "sigmas": ["15", 0],
            "latent_image": ["12", 0],
        },
    },
    "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
    "18": {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["17", 0],
            "vae": ["3", 0],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 16,
        },
    },
    "19": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["17", 1], "audio_vae": ["4", 0]}},
    "23": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": _UPSCALE_MODEL}},
    "24": {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["23", 0], "image": ["18", 0]},
    },
    "26": {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["24", 0],
            "upscale_method": "nearest-exact",
            "width": _VID_CROP_16_9_2X_W,
            "height": _VID_CROP_16_9_2X_H,
            "crop": "center",
        },
    },
    "25": {"class_type": "CreateVideo", "inputs": {"images": ["26", 0], "fps": 24, "audio": ["19", 0]}},
    "30": {
        "class_type": "SaveVideo",
        "inputs": {"video": ["25", 0], "filename_prefix": "media_gen_t2v", "format": "auto"},
    },
}

_T2I = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": _QWEN_UNET, "weight_dtype": "default"}},
    "3": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": _QWEN_CLIP,
            "type": "qwen_image",
            "device": "default",
        },
    },
    "4": {"class_type": "VAELoader", "inputs": {"vae_name": _QWEN_VAE}},
    "5": {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
            "clip": ["3", 0],
            "prompt": "",
            "negative_prompt": _T2I_NEGATIVE,
            "resolution": _T2I_BASE_16_9_W,
        },
    },
    "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": _T2I_BASE_16_9_W, "height": _T2I_BASE_16_9_H, "batch_size": 1},
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": 0,
            "steps": 25,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["5", 0],
            "negative": ["5", 1],
            "latent_image": ["6", 0],
            "denoise": 1.0,
        },
    },
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}},
    "22": {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["8", 0]}},
    "23": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": _UPSCALE_MODEL}},
    "24": {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["23", 0], "image": ["22", 0]},
    },
    "26": {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["24", 0],
            "upscale_method": "nearest-exact",
            "width": _T2I_BASE_16_9_W * 2,
            "height": _T2I_BASE_16_9_H * 2,
            "crop": "center",
        },
    },
    "30": {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": "media_gen_t2i"}},
}

_EDIT = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": _QWEN_UNET, "weight_dtype": "default"}},
    "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
    "3": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": _QWEN_CLIP,
            "type": "qwen_image",
            "device": "default",
        },
    },
    "4": {"class_type": "VAELoader", "inputs": {"vae_name": _QWEN_VAE}},
    "6": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "33": {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["6", 0],
            "upscale_method": _EDIT_SCALE_METHOD,
            "width": 0,
            "height": 0,
            "crop": "disabled",
        },
    },
    "8": {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
            "clip": ["3", 0],
            "prompt": "",
            "negative_prompt": _T2I_NEGATIVE,
            "vae": ["4", 0],
            "resolution": 0,
            "images.image_1": ["33", 0],
        },
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["2", 0],
            "seed": 0,
            "steps": 25,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["8", 0],
            "negative": ["8", 1],
            "latent_image": ["8", 2],
            "denoise": 1.0,
        },
    },
    "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}},
    "22": {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["9", 0]}},
    "30": {"class_type": "SaveImage", "inputs": {"images": ["22", 0], "filename_prefix": "media_gen_edit"}},
}


def build_t2v_prompt(prompt: str, seed: int, aspect_ratio: str = "16:9", upscale: bool = False) -> dict:
    """A prompt for the LTX-2.5 text to video pipeline.

    Args:
        prompt: The text to animate.
        seed: The seed for the noise.
        aspect_ratio: The output aspect ratio.
        upscale: Apply the 2x upscaler before the center crop.

    Returns:
        A fresh copy of the template with the prompt, the seed, and the
        aspect-ratio dimensions filled in.
    """
    api = copy.deepcopy(_T2V)
    _set_video_models(api)
    api["5"]["inputs"]["text"] = prompt
    api["13"]["inputs"]["noise_seed"] = seed
    base_w, base_h = _video_base(aspect_ratio)
    api["10"]["inputs"]["width"] = base_w
    api["10"]["inputs"]["height"] = base_h
    _set_crop_tail(api, "18", _crop_target(_VIDEO_CROP_TARGETS, aspect_ratio, upscale), upscale)
    return api


def build_i2v_prompt(prompt: str, image: str, seed: int, aspect_ratio: str = "16:9", upscale: bool = False) -> dict:
    """A prompt for the LTX-2.5 image to video pipeline.

    Args:
        prompt: The text to animate.
        image: The public URL of the first frame.
        seed: The seed for the noise.
        aspect_ratio: The output aspect ratio.
        upscale: Apply the 2x upscaler before the center crop.

    Returns:
        A fresh copy of the template with the prompt, the image, the seed, and
        the aspect-ratio dimensions filled in.
    """
    # The t2v wiring with the image branch added. The image enters at node 22,
    # and node 12 then reads the video latent from it.
    api = build_t2v_prompt(prompt, seed, aspect_ratio, upscale)
    api["12"]["inputs"]["video_latent"] = ["22", 0]
    api["30"]["inputs"]["filename_prefix"] = "media_gen_i2v"
    api["20"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    api["21"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": _IMG_COMPRESSION}}
    api["22"] = {
        "class_type": "LTXVImgToVideoInplace",
        "inputs": {"vae": ["3", 0], "image": ["21", 0], "latent": ["10", 0], "strength": _GUIDE_STRENGTH, "bypass": False},
    }
    return api


def build_t2i_prompt(prompt: str, seed: int, aspect_ratio: str = "16:9", upscale: bool = False) -> dict:
    """A prompt for the Qwen-Image-2.1 text to image pipeline.

    Args:
        prompt: The text to paint.
        seed: The seed for the sampler.
        aspect_ratio: The output aspect ratio.
        upscale: Apply the 2x upscaler before the center crop.

    Returns:
        A fresh copy of the template with the prompt, the seed, and the
        aspect-ratio dimensions filled in.
    """
    api = copy.deepcopy(_T2I)
    _set_image_models(api)
    api["5"]["inputs"]["prompt"] = prompt
    api["7"]["inputs"]["seed"] = seed
    base_w, base_h = _t2i_base(aspect_ratio)
    api["5"]["inputs"]["resolution"] = max(base_w, base_h)
    api["6"]["inputs"]["width"] = base_w
    api["6"]["inputs"]["height"] = base_h
    _set_crop_tail(api, "22", _crop_target(_T2I_CROP_TARGETS, aspect_ratio, upscale), upscale)
    return api


def _edit_scale_node(source_node: str, width: int, height: int) -> dict:
    """An ImageScale node that fits a loaded image to the exact target pixels."""
    return {
        "class_type": "ImageScale",
        "inputs": {
            "image": [source_node, 0],
            "upscale_method": _EDIT_SCALE_METHOD,
            "width": width,
            "height": height,
            "crop": "disabled",
        },
    }


def build_edit_prompt(
    instruction: str,
    image: str,
    seed: int,
    image_dims: tuple[int, int],
    extra_images: Sequence[str] = (),
    extra_image_dims: Sequence[tuple[int, int]] = (),
) -> dict:
    """A prompt for the Qwen-Image-2.1 image edit pipeline.

    Args:
        instruction: The edit to apply to the image.
        image: The public URL of the source image.
        seed: The seed for the sampler.
        image_dims: The fitted canvas pixels the scale node targets.
        extra_images: Up to two more reference URLs.
        extra_image_dims: The fitted pixels per reference, in order.

    Returns:
        A fresh copy of the template with the instruction, the images, and
        the seed filled in.

    Raises:
        ValueError: If more than two extra images are given.
    """
    if len(extra_images) > len(_EXTRA_IMAGE_NODE_IDS):
        raise ValueError(
            f"the edit pipeline takes at most {len(_EXTRA_IMAGE_NODE_IDS) + 1} "
            f"reference images, got {len(extra_images) + 1}"
        )
    api = copy.deepcopy(_EDIT)
    _set_image_models(api)
    api["8"]["inputs"]["prompt"] = instruction
    api["6"]["inputs"]["image"] = image
    api["7"]["inputs"]["seed"] = seed
    canvas_width, canvas_height = image_dims
    api[_EDIT_SCALE_NODE_ID]["inputs"]["width"] = canvas_width
    api[_EDIT_SCALE_NODE_ID]["inputs"]["height"] = canvas_height
    for index, url in enumerate(extra_images):
        node_id = _EXTRA_IMAGE_NODE_IDS[index]
        scale_node_id = _EXTRA_IMAGE_SCALE_NODE_IDS[index]
        width, height = extra_image_dims[index]
        api[node_id] = {"class_type": "LoadImage", "inputs": {"image": url}}
        api[scale_node_id] = _edit_scale_node(node_id, width, height)
        api["8"]["inputs"][f"images.image_{index + 2}"] = [scale_node_id, 0]
    return api


def build_f2lv_prompt(
    prompt: str,
    first_frame: str,
    last_frame: str,
    seed: int,
    aspect_ratio: str = "16:9",
    upscale: bool = False,
) -> dict:
    """A prompt for the LTX-2.5 first and last frame to video pipeline.

    Args:
        prompt: The text to animate.
        first_frame: The public URL of the first frame.
        last_frame: The public URL of the last frame.
        seed: The seed for the noise.
        aspect_ratio: The output aspect ratio.
        upscale: Apply the 2x upscaler before the center crop.

    Returns:
        A fresh copy of the template with the frames, the seed, and the
        aspect-ratio dimensions filled in.
    """
    api = build_t2v_prompt(prompt, seed, aspect_ratio, upscale)
    api["40"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
    api["41"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
    api["42"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["40", 0], "img_compression": _IMG_COMPRESSION}}
    api["43"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["41", 0], "img_compression": _IMG_COMPRESSION}}
    api["44"] = {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": ["7", 0],
            "negative": ["7", 1],
            "vae": ["3", 0],
            "latent": ["10", 0],
            "image": ["42", 0],
            "frame_idx": 0,
            "strength": _GUIDE_STRENGTH,
        },
    }
    api["45"] = {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": ["44", 0],
            "negative": ["44", 1],
            "vae": ["3", 0],
            "latent": ["44", 2],
            "image": ["43", 0],
            "frame_idx": -1,
            "strength": _GUIDE_STRENGTH,
        },
    }
    api["46"] = {
        "class_type": "LTXVCropGuides",
        "inputs": {"positive": ["45", 0], "negative": ["45", 1], "latent": ["17", 0]},
    }
    api["8"]["inputs"]["positive"] = ["45", 0]
    api["8"]["inputs"]["negative"] = ["45", 1]
    api["12"]["inputs"]["video_latent"] = ["45", 2]
    api["18"]["inputs"]["samples"] = ["46", 2]
    api["30"]["inputs"]["filename_prefix"] = "media_gen_f2lv"
    return api
