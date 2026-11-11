"""Describe what happens in a video file."""

import asyncio
import os
import sys
import json
import uuid
import shutil
from typing import Optional, Tuple

from app.lib.cli.cli_runner import run_cli
from app.lib.reason.reason import reason
from app.lib.shared import paths

__all__ = ["describe_video"]

_SCRATCH_NAME = "video_frames"
_BATCH_SIZE = 5
_FRAME_INTERVAL = 2.0
_MAX_FRAMES = 60
_MIN_FRAMES = 5
_FRAME_HEIGHT = 720
_FRAME_QUALITY = 2

_NATIVE_VIDEO_PROMPT = """First determine whether you can actually inspect the supplied video.
If the video is missing, ignored, unsupported, inaccessible, or cannot be decoded, return {"success": false} with no description.
Set success to true only if you actually inspected the video. Never infer its contents from the prompt, context, filename, or general knowledge.
On success, return {"success": true, "description": "..."}, with a natural, cohesive summary of the observed video in chronological order.
CRITICAL INSTRUCTIONS:
1. SUBJECT-CENTRIC NARRATIVE: Focus on the subjects and their actions, not the camera.
2. CAMERA MOVEMENT: Distinguish camera movement (panning, tracking, zooming, handheld shake) from subject movement. If the background shifts while the subject's posture stays unchanged, it is camera motion.
3. IDENTITY CONSISTENCY: Maintain consistent subject labels throughout the summary.
4. ON-SCREEN TEXT: Transcribe any visible text, numbers, or signage verbatim.
5. AUDIO: Incorporate speech or other significant audio when present.
6. OUTPUT FORMAT: Output only the JSON object. Include a nonempty description only when success is true. Write the description in a neutral, objective tone, without conversational filler, greetings, or closing remarks.
"""


def _parse_native_description(response: str) -> str | None:
    """Read the description from a successful video reply."""
    try:
        result = json.loads(response)
    except (json.JSONDecodeError, TypeError, RecursionError):
        return None
    if not isinstance(result, dict) or result.get("success") is not True:
        return None
    if set(result) != {"success", "description"}:
        return None
    description = result["description"]
    if not isinstance(description, str) or not description.strip():
        return None
    return description.strip()


async def _get_video_duration(video_path: str) -> float:
    """Find how long a video is.

    Args:
        video_path: Path to the video file.

    Returns:
        The length in seconds, or -1 when it cannot be read.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = await run_cli(cmd, "ffprobe duration")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "ffprobe failed")
        return float(result.stdout)
    except Exception as e:
        print(f"Error getting video duration: {e}", file=sys.stderr)
        return -1.0


async def _get_metadata_context(video_path: str) -> Optional[str]:
    """Read the extra info stored inside a video, such as the device used.

    Args:
        video_path: Path to the video file.

    Returns:
        The info as JSON text, or None when it cannot be read.
    """
    try:
        cmd = ["exiftool", "-j", "-G", video_path]
        result = await run_cli(cmd, "exiftool metadata extraction")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "exiftool failed")
        return await asyncio.to_thread(_parse_metadata_json, result.stdout)
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"Error extracting metadata: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_metadata_context: {e}", file=sys.stderr)
        return None


def _parse_metadata_json(stdout: str) -> str:
    """Pick the wanted fields out of a video info dump.

    This blocks, so callers must run it off the event loop.

    Args:
        stdout: The raw info dump as JSON text.

    Returns:
        The picked fields as JSON text.
    """
    raw_data = json.loads(stdout)[0]
    metadata = {}
    for k in [
        "EXIF:Make", "EXIF:Model", "EXIF:Software",
        "QuickTime:Make", "QuickTime:Model", "QuickTime:Software",
        "MakerNotes:Device", "Keys:Model", "Keys:Make",
        "QuickTime:HandlerDescription",
    ]:
        if k in raw_data:
            metadata[k] = raw_data[k]

    if any(marker in str(raw_data) for marker in ["LivePhoto", "TopShot", "MicroVideo"]):
        metadata["CaptureType"] = "Motion Photo / Live Photo"
    return json.dumps(metadata)


async def _extract_frame(video_path: str, timestamp: float, output_path: str) -> Tuple[bool, int]:
    """Save one picture from a video at a given time.

    Args:
        video_path: Path to the video file.
        timestamp: How many seconds into the video to grab.
        output_path: Where the picture is written.

    Returns:
        Whether it worked and a status code. The code is 0 when it worked
        and 1 when it did not.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(timestamp),
        "-vf", f"scale=-2:{_FRAME_HEIGHT}",
        "-frames:v", "1",
        "-q:v", str(_FRAME_QUALITY),
        output_path,
        "-y", "-loglevel", "error",
    ]
    try:
        result = await run_cli(cmd, "ffmpeg frame extraction")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "ffmpeg failed")
        return True, 0
    except Exception as e:
        print(f"Error extracting frame: {e}", file=sys.stderr)
        return False, 1


async def _summarize_descriptions(
    descriptions: list[str], context: str, metadata: str
) -> Tuple[str, int]:
    """Turn descriptions of the video pictures into one summary.

    Args:
        descriptions: What each batch of pictures showed, in time order.
        context: Extra context for the model.
        metadata: The extra info stored inside the video, as JSON text.

    Returns:
        The summary and a status code. The code is 0 when it worked and 1
        when it did not.
    """
    combined = "\n\n".join(
        f"Frame Batch {i + 1}:\n{desc}" for i, desc in enumerate(descriptions)
    )
    summary_prompt = f"""You are an AI assistant creating a cohesive video summary based on literal frame-batch descriptions.
Raw Video Metadata (EXIF): {metadata}

Below are literal descriptions of batches of frames extracted from the video in CHRONOLOGICAL ORDER.
{combined}

Based on these descriptions, provide a natural, cohesive, and engaging summary.

CRITICAL INSTRUCTIONS:
1. SUBJECT-CENTRIC NARRATIVE: Focus the summary on the subjects and their actions. Most people are interested in what the subject is doing, not the camera.
2. CAMERA MOVEMENT AS INTERNAL CONTEXT: Use information about camera movement (panning, tracking, zooming) to accurately understand what the subjects are doing (e.g. to avoid mislabeling a camera pan as a subject walking), but DO NOT over-describe the camera movements in the final summary. The focus should stay on the story, not the cinematography.
3. DEVICE AWARENESS: Analyze the 'Raw Video Metadata' above. If it indicates a phone or handheld device, be extremely skeptical of reported subject movement. Videos recorded by phone tend to move angles and positions constantly; assume shifting backgrounds are camera motion rather than subject movement unless you see clear subject intent (e.g. stepping).
4. IDENTITY CONSISTENCY: Maintain consistent subject labels.
5. LITERAL TO NATURAL: Translate mechanical batch notes into a human-readable story. Maintain correct chronological order.
6. OUTPUT FORMAT: No conversational filler, no greetings, no 'here is the description,' and no closing remarks. Output only the data requested in a neutral, objective tone.
"""
    return await reason(summary_prompt, context, thinking=False)


async def _check_dependencies() -> Tuple[Optional[str], Optional[int]]:
    """Check that the video tools this module needs are installed.

    Returns:
        An error message and status code 3 when a tool is missing, or two
        Nones when they are all there.
    """
    for cmd in ("ffprobe", "ffmpeg"):
        if await asyncio.to_thread(shutil.which, cmd) is None:
            return f"{cmd} not found. Please install FFmpeg.", 3
    return None, None


async def _process_frames(
    timestamps: list[float],
    num_frames: int,
    video_path: str,
    context: str,
    metadata_json: str,
    temp_dir: str,
    verbose: bool,
) -> Tuple[str, int]:
    """Describe pictures taken from a video and sum them up.

    Args:
        timestamps: How many seconds into the video each picture comes from.
        num_frames: How many pictures there are in all.
        video_path: Path to the video file.
        context: Extra context for the model.
        metadata_json: The extra info stored inside the video, as JSON text.
        temp_dir: Folder the pictures are written to.
        verbose: If True, prints progress messages.

    Returns:
        The summary and a status code. The code is 0 when it worked and 1
        when a picture or a description could not be made.
    """
    frame_descriptions = []

    for i in range(0, len(timestamps), _BATCH_SIZE):
        batch_timestamps = timestamps[i : i + _BATCH_SIZE]
        batch_paths = []

        for j, ts in enumerate(batch_timestamps):
            frame_path = os.path.join(temp_dir, f"frame_{i + j}.jpg")
            success, _status = await _extract_frame(video_path, ts, frame_path)
            if success:
                batch_paths.append(frame_path)
            else:
                return f"Could not extract frame at timestamp {ts}.", 1

        desc, status = await _analyze_batch(
            batch_paths, i, num_frames, context, metadata_json, frame_descriptions
        )
        if status != 0:
            return desc, status
        frame_descriptions.append(desc)

        if verbose:
            print(desc)
            print(
                f"Processed {min(i + _BATCH_SIZE, num_frames)}/{num_frames} frames..."
            )

    if not frame_descriptions:
        return "Could not describe any frames from the video.", 1
    return await _summarize_descriptions(frame_descriptions, context, metadata_json)


async def _analyze_batch(
    batch_paths: list[str],
    batch_start: int,
    num_frames: int,
    context: str,
    metadata_json: str,
    frame_descriptions: list[str],
) -> Tuple[str, int]:
    """Describe one batch of pictures from a video.

    Args:
        batch_paths: Paths to the pictures in this batch.
        batch_start: Where this batch starts in the whole run.
        num_frames: How many pictures there are in all.
        context: Extra context for the model.
        metadata_json: The extra info stored inside the video, as JSON text.
        frame_descriptions: What the earlier batches showed.

    Returns:
        The description and a status code. The code is 0 when it worked and
        1 when it did not.
    """
    recent_desc = "\n\n".join(frame_descriptions[-2:]) if frame_descriptions else ""
    batch_context = (
        f"{context}\n\nMetadata: {metadata_json}\n\nPrevious frames show: {recent_desc}"
        if recent_desc
        else f"{context}\n\nMetadata: {metadata_json}"
    )

    frame_prompt = f"""Analyze these {len(batch_paths)} consecutive video frames (frame {batch_start + 1}-{batch_start + _BATCH_SIZE}) from a {num_frames} frames video sample. You are looking at the batch number {batch_start // _BATCH_SIZE + 1} from the total {num_frames // _BATCH_SIZE} batches. Provide a strictly literal description.
CRITICAL MOTION ATTRIBUTION: Check the 'Metadata' provided. If it indicates a phone/mobile device, you must carefully distinguish between subject motion and camera movement (panning, tilting, shaking). Videos recorded by phone tend to move angles and positions constantly. If the background/floor is sliding but the subject's relative posture (legs, body angle) is unchanging, it is CAMERA MOTION.
OUTPUT FORMAT: No conversational filler, no greetings, no 'here is the description,' and no closing remarks. Output only the data requested in a neutral, objective tone.
"""
    return await reason(frame_prompt, batch_context, thinking=False, image_files=batch_paths)


async def describe_video(video_path: str, context: str = "", verbose: bool = False) -> Tuple[str, int]:
    """Describe a video file.

    Args:
        video_path: Path to the video file.
        context: Extra context for the model.
        verbose: If True, prints progress messages.

    Returns:
        The description and a status code. The code is 0 when it worked, 1
        when the video could not be read, 2 when the file is missing, 3 when
        a needed tool is not installed, and 4 when the length is unknown.
    """
    if not await asyncio.to_thread(os.path.exists, video_path):
        return f"Video path '{video_path}' does not exist.", 2

    native_response, native_status = await reason(
        _NATIVE_VIDEO_PROMPT, context, thinking=False,
        json_format=True, video_files=[video_path],
    )
    if native_status == 0:
        native_desc = await asyncio.to_thread(_parse_native_description, native_response)
        if native_desc is not None:
            return native_desc, 0

    return await _describe_video_by_frames(video_path, context, verbose)


async def _describe_video_by_frames(
    video_path: str, context: str, verbose: bool
) -> Tuple[str, int]:
    """Describe a video from pictures taken across it.

    The pictures are deleted before returning.

    Args:
        video_path: Path to the video file.
        context: Extra context for the model.
        verbose: If True, prints progress messages.

    Returns:
        The description and a status code. The code is 0 when it worked, 1
        when the video could not be read, 3 when a needed tool is not
        installed, and 4 when the length is unknown.
    """
    error_msg, error_code = await _check_dependencies()
    if error_msg:
        return error_msg, error_code

    duration = await _get_video_duration(video_path)
    if duration <= 0:
        return "Failed to parse video duration.", 4
    metadata_json = await _get_metadata_context(video_path)
    if metadata_json is None:
        return "Could not extract video metadata.", 1

    num_frames = min(_MAX_FRAMES, max(_MIN_FRAMES, int(duration // _FRAME_INTERVAL)))
    timestamps = [
        (duration / (num_frames + 1)) * (i + 1) for i in range(num_frames)
    ]

    run_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(paths.scratch_dir(_SCRATCH_NAME), f"video_frames_{run_id}")
    await asyncio.to_thread(os.makedirs, temp_dir, exist_ok=True)

    if verbose:
        print(f"Processing video: {num_frames} frames...")

    try:
        return await _process_frames(
            timestamps, num_frames, video_path,
            context, metadata_json, temp_dir, verbose,
        )
    finally:
        await asyncio.to_thread(shutil.rmtree, temp_dir)
