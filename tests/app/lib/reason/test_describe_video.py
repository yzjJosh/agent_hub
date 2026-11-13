"""Tests for app.lib.reason.describe_video module."""

import os
import shutil
from pathlib import Path
from typing import Callable, Self
from unittest.mock import patch, MagicMock

import httpx
import pytest
from PIL import Image
from app.lib.reason.describe_video import describe_video
from tests.app.conftest import is_llm_available
from tests.app.lib.conftest import mock_cli_proc

_DURATION_10S = 10.0
_DURATION_2S = 2.0


def _mock_cli_side_effect(
    duration: float,
    metadata: str = "[{}]",
    extract_success: bool = True,
    extract_error: bool = False,
    metadata_error: bool = False,
    call_count: list[int] | None = None,
) -> Callable[..., MagicMock]:
    """Build a stand in for the tools describe_video runs.

    Args:
        duration: How long the video is said to be.
        metadata: Metadata text for the video.
        extract_success: Whether frame files are really written.
        extract_error: Whether pulling out frames fails.
        metadata_error: Whether reading the metadata fails.
        call_count: One item list that counts the frame calls.

    Returns:
        A stand in for starting a child process.
    """
    def _side_effect(*cmd: str, **_kw: int) -> MagicMock:
        if cmd[0] == "ffprobe":
            return mock_cli_proc(stdout=f"{duration}\n")
        if cmd[0] == "exiftool":
            if "-MimeType" in cmd:
                mime_type = "video/mp4" if cmd[-1].endswith(".mp4") else "image/jpeg"
                return mock_cli_proc(stdout=mime_type)
            if metadata_error:
                return mock_cli_proc(stderr="exiftool failed", returncode=1)
            return mock_cli_proc(stdout=metadata)
        if cmd[0] == "ffmpeg":
            if call_count is not None:
                call_count[0] += 1
            if extract_error:
                return mock_cli_proc(stderr="ffmpeg failed", returncode=1)
            if extract_success:
                output_idx = cmd.index("-y") - 1
                output_path = cmd[output_idx]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                Image.new("RGB", (32, 32), "red").save(output_path)
            return mock_cli_proc()
        return mock_cli_proc()
    return _side_effect


def _reason_fallback_side_effect(*_args: str, **kwargs: bool | list[str]) -> tuple[str, int]:
    """Fail when a whole video is sent and give a description otherwise.

    Returns:
        The result and status.
    """
    if "video_files" in kwargs:
        return ("Native video unsupported", 1)
    return ("test description", 0)


def _reason_native_failure(*_args: str, **_kwargs: bool | list[str]) -> tuple[str, int]:
    """Always fail the model call."""
    return ("Native video unsupported", 1)


def _chat_response(content: str, status_code: int = 200) -> httpx.Response:
    """Build a reply from the model server."""
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://video.test/v1/chat/completions"),
        json={"choices": [{"message": {"content": content}}]},
    )


@pytest.fixture
def dummy_video(tmp_path: Path) -> str:
    """Create a small valid MP4 file."""
    mp4_content = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x01isomiso2avc1mp41"
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(mp4_content)
    return str(video_path)


class TestDescribeVideoErrors:
    """Test describe_video error handling."""

    async def test_returns_error_for_nonexistent_file(self: Self) -> None:
        """A missing file gives status 2."""
        result, status = await describe_video("/nonexistent/path/file.mp4")

        assert status == 2
        assert "does not exist" in result

    async def test_returns_error_for_missing_ffprobe(self, dummy_video: str) -> None:
        """A missing ffprobe tool gives status 3 and names the tool."""
        with patch('app.lib.reason.describe_video.shutil.which') as mock_which:
            mock_which.side_effect = (
                lambda cmd: None if cmd == "ffprobe" else "/usr/bin/" + cmd
            )
            with patch('app.lib.reason.describe_video.reason', side_effect=_reason_native_failure):
                result, status = await describe_video(dummy_video)

        assert status == 3
        assert "ffprobe" in result

    async def test_returns_error_for_missing_ffmpeg(self, dummy_video: str) -> None:
        """A missing ffmpeg tool gives status 3 and names the tool."""
        with patch('app.lib.reason.describe_video.shutil.which') as mock_which:
            mock_which.side_effect = (
                lambda cmd: None if cmd == "ffmpeg" else "/usr/bin/" + cmd
            )
            with patch('app.lib.reason.describe_video.reason', side_effect=_reason_native_failure):
                result, status = await describe_video(dummy_video)

        assert status == 3
        assert "ffmpeg" in result

    async def test_returns_error_for_zero_duration(self, dummy_video: str) -> None:
        """A video with no length gives status 4."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.return_value = mock_cli_proc(stdout="0\n")
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_native_failure):
                    result, status = await describe_video(dummy_video)

        assert status == 4
        assert "duration" in result.lower()

    async def test_returns_error_for_invalid_duration(self, dummy_video: str) -> None:
        """A video whose length cannot be read gives status 4."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = lambda *_a, **_kw: mock_cli_proc(stderr="ffprobe failed", returncode=1)
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_native_failure):
                    result, status = await describe_video(dummy_video)

        assert status == 4
        assert "duration" in result.lower()

    async def test_returns_error_for_missing_metadata(self, dummy_video: str) -> None:
        """A video whose metadata cannot be read gives status 1."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(
                    _DURATION_10S, metadata_error=True
                )
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_native_failure):
                    with patch('app.lib.reason.describe_video.shutil.rmtree'):
                        result, status = await describe_video(dummy_video)

        assert status == 1
        assert "metadata" in result.lower()


class TestDescribeVideoNative:
    """Test sending the whole video to the model."""

    async def test_native_success_skips_frame_extraction(
        self, dummy_video: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the model takes the whole video, no frames are pulled out."""
        monkeypatch.setenv("REASON_BACKEND_URL", "http://video.test")
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value = _chat_response(
                '{"success": true, "description": "  native summary  "}'
            )
            with patch('asyncio.create_subprocess_exec') as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S)
                result, status = await describe_video(dummy_video)

        assert status == 0
        assert result == "native summary"
        assert [call.args[0] for call in mock_run.call_args_list] == ["exiftool"]
        mock_post.assert_awaited_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}
        content = payload["messages"][1]["content"]
        assert content[1]["type"] == "video_url"
        assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert '"description": "..."' in content[0]["text"]
        assert "User question" not in content[0]["text"]
        assert '{"success": false}' in content[0]["text"]
        assert "actually inspected the video" in content[0]["text"]

    @pytest.mark.parametrize("native_content", [
        '{"success": false}',
        '{"success": false, "description": "Invented video contents"}',
        'I cannot view videos, but here is a possible description.',
        '{"success": true,',
        'null',
        '[]',
        '"native summary"',
        '{}',
        '{"description": "Missing success"}',
        '{"success": "false", "description": "Invented video contents"}',
        '{"success": "true", "description": "Invented video contents"}',
        '{"success": 1, "description": "Invented video contents"}',
        '{"success": null, "description": "Invented video contents"}',
        '{"success": true}',
        '{"success": true, "description": ""}',
        '{"success": true, "description": "   "}',
        '{"success": true, "description": null}',
        '{"success": true, "description": 123}',
        '{"success": true, "description": "summary", "extra": "field"}',
    ])
    async def test_native_failure_falls_back_to_frames(
        self, dummy_video: str, native_content: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unconfirmed video replies are replaced by descriptions from frames."""
        monkeypatch.setenv("REASON_BACKEND_URL", "http://video.test")
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S)
                with patch('httpx.AsyncClient.post') as mock_post:
                    mock_post.side_effect = [
                        _chat_response(native_content),
                        _chat_response("Observed red frames"),
                        _chat_response("Summary of observed red frames"),
                    ]
                    result, status = await describe_video(dummy_video)

        assert status == 0
        assert result == "Summary of observed red frames"
        assert mock_post.await_count == 3
        native_payload, frames_payload, summary_payload = [
            call.kwargs["json"] for call in mock_post.call_args_list
        ]
        assert native_payload["messages"][1]["content"][1]["type"] == "video_url"
        frames_content = frames_payload["messages"][1]["content"]
        assert [part["type"] for part in frames_content[1:]] == ["image_url"] * 5
        summary_content = summary_payload["messages"][1]["content"]
        assert "Observed red frames" in summary_content
        assert "response_format" not in frames_payload
        assert "response_format" not in summary_payload

    @pytest.mark.parametrize("native_status", [200, 500])
    async def test_frame_failure_after_native_failure(
        self, dummy_video: str, native_status: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Frame request failures still reach the caller after native failure."""
        monkeypatch.setenv("REASON_BACKEND_URL", "http://video.test")
        with patch('shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec') as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S)
                with patch('httpx.AsyncClient.post') as mock_post:
                    mock_post.side_effect = [
                        _chat_response('{"success": false}', native_status),
                        _chat_response("Backend unavailable", 503),
                    ]
                    result, status = await describe_video(dummy_video)

        assert status == 1
        assert "503" in result
        assert mock_post.await_count == 2


class TestDescribeVideoQuestion:
    """Test asking a question about the whole video."""

    async def test_empty_context_asks_for_description(self, dummy_video: str) -> None:
        """With no question, the model is asked to describe the video."""
        with patch('app.lib.reason.describe_video.reason') as mock_reason:
            mock_reason.return_value = ('{"success": true, "description": " A summary "}', 0)
            result, status = await describe_video(dummy_video)

        assert (result, status) == ("A summary", 0)
        mock_reason.assert_awaited_once()
        prompt, context = mock_reason.call_args.args
        assert '"description": "..."' in prompt
        assert '"answer"' not in prompt
        assert context == ""
        assert mock_reason.call_args.kwargs["video_files"] == [dummy_video]
        assert mock_reason.call_args.kwargs["json_format"] is True

    async def test_question_is_sent_with_video(self, dummy_video: str) -> None:
        """A question is asked about the video and the answer comes back."""
        question = "At what time does the yellow square appear?"
        with patch('app.lib.reason.describe_video.reason') as mock_reason:
            mock_reason.return_value = (
                '{"success": true, "answer": " Between 74 and 80 seconds. "}', 0
            )
            result, status = await describe_video(dummy_video, f"  {question}  ")

        assert (result, status) == ("Between 74 and 80 seconds.", 0)
        mock_reason.assert_awaited_once()
        prompt, context = mock_reason.call_args.args
        assert prompt.endswith(f"User question: {question}")
        assert '"answer": "..."' in prompt
        assert '"description"' not in prompt
        assert context == ""
        assert mock_reason.call_args.kwargs["video_files"] == [dummy_video]
        assert mock_reason.call_args.kwargs["json_format"] is True

    @pytest.mark.parametrize("native_reply", [
        ('{"success": false}', 0),
        ('{"success": true, "description": "A general summary"}', 0),
        ('{"success": true, "answer": ""}', 0),
        ('{"success": true, "answer": 74}', 0),
        ('{"success": "true", "answer": "Invented answer"}', 0),
        ('{"success": true, "answer": "74 seconds", "extra": "field"}', 0),
        ('The square appears at 74 seconds.', 0),
        ("Backend unavailable", 1),
    ])
    async def test_bad_answer_falls_back_to_frames(
        self, dummy_video: str, native_reply: tuple[str, int],
    ) -> None:
        """An answer that is not a clean success is replaced by one from frames."""
        question = "When does the yellow square appear?"

        def _reason_side_effect(*_args: str, **kwargs: bool | list[str]) -> tuple[str, int]:
            if "video_files" in kwargs:
                return native_reply
            return ("Frame answer", 0)

        with patch('shutil.which', return_value="/usr/bin/tool"):
            with patch('asyncio.create_subprocess_exec') as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S)
                with patch(
                    'app.lib.reason.describe_video.reason', side_effect=_reason_side_effect,
                ) as mock_reason:
                    result, status = await describe_video(dummy_video, question)

        assert (result, status) == ("Frame answer", 0)
        native_call, *frame_calls = mock_reason.call_args_list
        assert native_call.kwargs["video_files"] == [dummy_video]
        assert len(frame_calls) == 2
        assert all(question in call.args[1] for call in frame_calls)


class TestDescribeVideoFrameCount:
    """Test describe_video with videos of different lengths."""

    async def test_short_video_minimum_frames(self, dummy_video: str) -> None:
        """A short video is still sent to the model."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_2S)
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_fallback_side_effect) as mock_reason:
                    with patch('app.lib.reason.describe_video.shutil.rmtree'):
                        await describe_video(dummy_video)

        mock_reason.assert_called()

    async def test_long_video_maximum_frames(self, dummy_video: str) -> None:
        """A long video has frames pulled out of it."""
        call_count = [0]
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(
                    300.0, call_count=call_count
                )
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_fallback_side_effect):
                    with patch('app.lib.reason.describe_video.shutil.rmtree'):
                        await describe_video(dummy_video)

        assert call_count[0] > 0


class TestDescribeVideoReasonFailure:
    """Test describe_video when the model fails."""

    async def test_returns_error_on_reason_failure(self, dummy_video: str) -> None:
        """A model failure on the frames gives status 1."""
        def _reason_side_effect(*_args: str, **kwargs: bool | list[str]) -> tuple[str, int]:
            if "video_files" in kwargs:
                return ("Native video unsupported", 1)
            return ("Backend unavailable", 1)

        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S)
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_side_effect):
                    with patch('app.lib.reason.describe_video.shutil.rmtree'):
                        result, status = await describe_video(dummy_video)

        assert status == 1
        assert "Backend" in result

    async def test_returns_error_on_frame_extraction_failure(self, dummy_video: str) -> None:
        """A failure pulling out frames gives status 1."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(
                    _DURATION_10S, extract_error=True
                )
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_fallback_side_effect):
                    with patch('app.lib.reason.describe_video.shutil.rmtree') as mock_rm:
                        result, status = await describe_video(dummy_video)

        assert status == 1
        assert "frame" in result.lower()
        mock_rm.assert_called_once()


class TestDescribeVideoCleanup:
    """Test that temp directories are cleaned up."""

    async def test_temp_dir_cleaned_on_success(self, dummy_video: str) -> None:
        """The temporary folder is removed after a good run."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S)
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_fallback_side_effect):
                    with patch(
                        'app.lib.reason.describe_video.shutil.rmtree'
                    ) as mock_rm:
                        await describe_video(dummy_video)

        mock_rm.assert_called_once()

    async def test_temp_dir_cleaned_on_error(self, dummy_video: str) -> None:
        """The temporary folder is removed after a failed run."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(
                    _DURATION_10S, extract_error=True
                )
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_fallback_side_effect):
                    with patch('app.lib.reason.describe_video.shutil.rmtree') as mock_rm:
                        await describe_video(dummy_video)

        mock_rm.assert_called_once()


class TestDescribeVideoMetadata:
    """Test describe_video with video metadata."""

    async def test_handles_empty_metadata(self, dummy_video: str) -> None:
        """A video with empty metadata is still sent to the model."""
        with patch(
            'app.lib.reason.describe_video.shutil.which', return_value="/usr/bin/tool"
        ):
            with patch(
                'asyncio.create_subprocess_exec'
            ) as mock_run:
                mock_run.side_effect = _mock_cli_side_effect(_DURATION_10S, metadata="[{}]")
                with patch('app.lib.reason.describe_video.reason', side_effect=_reason_fallback_side_effect) as mock_reason:
                    with patch('app.lib.reason.describe_video.shutil.rmtree'):
                        await describe_video(dummy_video)

        mock_reason.assert_called()


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@pytest.mark.live
class TestDescribeVideoLive:
    """Live tests with a real model server."""

    def _copy_test_video(self: Self, tmp_path: Path, data_dir: str) -> str:
        """Copy a test video to tmp_path and return path."""
        src = os.path.join(data_dir, "test_video.mp4")
        dst = tmp_path / "test_video.mp4"
        shutil.copy(src, dst)
        return str(dst)

    @pytest.mark.timeout(120)
    async def test_describe_video(self, tmp_path: Path, data_dir: str) -> None:
        """A real video gets a description."""
        video_path = self._copy_test_video(tmp_path, data_dir)
        result, status = await describe_video(video_path)

        assert status == 0
        assert "tablet" in result.lower()

    @pytest.mark.timeout(120)
    async def test_describe_video_answers_question(self, tmp_path: Path, data_dir: str) -> None:
        """A real video gets an answer to a question about it."""
        video_path = self._copy_test_video(tmp_path, data_dir)
        result, status = await describe_video(
            video_path, "Which product name is shown at the end of this video?"
        )

        assert status == 0
        assert "chromecast" in result.lower()
