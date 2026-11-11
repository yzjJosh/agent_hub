"""Tests for app.tools.read_media_or_doc."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.lib.resolver.resolver import ResolveError
from app.lib.tool.truncation import DEFAULT_TOOL_OUTPUT_LEN_LIMIT
from app.tools.read_media_or_doc import read_media_or_doc_tool
from tests.app.lib.tool.conftest import assert_max_char_wire_contract, run_text

_OVERFLOW = 1_000
_AUDIO_FIXTURE = Path(__file__).parent.parent / "lib" / "data" / "test_audio.wav"


def _point_resources_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the resources folder at a temp folder."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))


def _share_file(tmp_path: Path, rel: str, data: bytes = b"fake media") -> Path:
    """Create a file in the resource share and return its absolute path."""
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestDocumentation:
    """Tests for what the tool and its params say."""

    async def test_max_char_docstring_uses_resolved_values(self) -> None:
        """The wire gives the real default and the real smallest value, and names no constant."""
        await assert_max_char_wire_contract(read_media_or_doc_tool, "read_media_or_doc")

    @pytest.mark.parametrize(
        "term",
        ["media or document", "resource", "web_fetch", "send_or_receive_file"],
    )
    def test_description_states_purpose_and_siblings(self, term: str) -> None:
        """The tool description says what it does and names the other file tools."""
        doc = read_media_or_doc_tool.description
        assert term in doc

    def test_resource_path_param_owns_listing_and_relative_path(self) -> None:
        """The base says the path is relative to the resource root, and the wire points at the listing tool."""
        p = next(p for p in read_media_or_doc_tool.params if p.name == "resource_path")
        assert "relative to the resource root" in p.description
        assert "resource listing tool" in p.wire_description
        assert "resources:///upload/video.mp4" in p.wire_description
        assert "send_or_receive_file" in p.wire_description
        assert "resources share" not in p.description
        assert "resources share" not in p.wire_description


class TestPromptParamDocs:
    """The prompt usage guidance lives on the prompt param, not the tool description."""

    def test_prompt_param_exists_optional_longtext(self) -> None:
        """The prompt param is present, optional, defaults to '', and is longtext."""
        p = next(p for p in read_media_or_doc_tool.params if p.name == "prompt")
        assert not p.required
        assert p.default == ""
        assert p.kind == "longtext"

    def test_prompt_param_carries_usage_guidance(self) -> None:
        """The prompt param carries the usage guidance and an example question."""
        doc = next(p.description for p in read_media_or_doc_tool.params if p.name == "prompt")
        assert "Set ONLY when you want the LLM to extract" in doc
        assert "What is the price of the product?" in doc

    def test_prompt_empty_default_gets_bare_optional_note(self) -> None:
        """The '' default adds the bare Optional note. The wire text is the base plus the note."""
        p = next(p for p in read_media_or_doc_tool.params if p.name == "prompt")
        assert p.wire_description == p.description + " Optional."

    def test_tool_description_has_no_prompt_guidance(self) -> None:
        """The tool description has no prompt usage guidance."""
        assert "Set the prompt parameter" not in read_media_or_doc_tool.description
        assert "prompt='What is the price" not in read_media_or_doc_tool.description


class TestResourcePathValidation:
    """Tests for bad resource paths."""

    async def test_missing_resource_points_to_upload_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path gives back upload help and reads nothing."""
        _point_resources_at(tmp_path, monkeypatch)
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            result = await run_text(read_media_or_doc_tool, "upload/nope.mp4")

        assert "Resource not found: upload/nope.mp4" in result
        assert "send_or_receive_file" in result
        mock_resolve.assert_not_called()

    async def test_path_escaping_share_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path that points outside the share is turned down and shows nothing."""
        _point_resources_at(tmp_path, monkeypatch)
        outside = tmp_path.parent / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret content")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            result = await run_text(read_media_or_doc_tool, "../outside/secret.txt")

        assert "Resource not found" in result
        assert "secret content" not in result
        mock_resolve.assert_not_called()

    async def test_directory_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A folder is turned down with a not a file error."""
        _point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            result = await run_text(read_media_or_doc_tool, "upload")

        assert "'upload' is a folder, not a file" in result
        mock_resolve.assert_not_called()


class TestConversion:
    """Tests for turning a file into text."""

    async def test_returns_converted_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file in the share comes back as text."""
        _point_resources_at(tmp_path, monkeypatch)
        media = _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = ("A lecture about AI.", False)
            result = await run_text(read_media_or_doc_tool, "upload/lecture.mp4")

        assert result == "A lecture about AI."
        mock_resolve.assert_called_once_with(str(media), context="", verbose=True)

    async def test_conversion_failure_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that cannot be turned into text gives back a failed tool message with the error text."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/clip.mp3")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.side_effect = ResolveError(
                "ASR transcription failed (HTTP 500): boom"
            )
            result = await run_text(read_media_or_doc_tool, "upload/clip.mp3")

        assert result.startswith("Tool 'read_media_or_doc' failed: ")
        assert "Error converting 'upload/clip.mp3' to text:" in result
        assert "ASR transcription failed (HTTP 500): boom" in result

    async def test_audio_without_asr_server_names_the_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An audio file with no ASR server set fails with a message that names the variable."""
        _point_resources_at(tmp_path, monkeypatch)
        monkeypatch.delenv("ASR_SERVER_URL", raising=False)
        _share_file(tmp_path, "upload/clip.wav", _AUDIO_FIXTURE.read_bytes())

        result = await run_text(read_media_or_doc_tool, "upload/clip.wav")

        assert result == (
            "Tool 'read_media_or_doc' failed: Error converting 'upload/clip.wav' to text: "
            "ASR_SERVER_URL is not set: needed by audio and video transcription"
        )
        assert (
            "[config] ASR_SERVER_URL is not set: needed by audio and video transcription\n"
            in capsys.readouterr().err
        )

    async def test_binary_download_notice_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A download notice for a binary file comes back unchanged."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/archive.zip")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            notice = (
                "Binary file with size 10 bytes. Please use WebDAV to download "
                "it instead:\n  curl -o /tmp/archive.zip http://localhost/dav/downloads/archive.zip"
            )
            mock_resolve.return_value = (notice, False)
            result = await run_text(read_media_or_doc_tool, "upload/archive.zip")

        assert result == notice


class TestTreeConversion:
    """Tests for turning page trees into markdown."""

    async def test_json_tree_converts_to_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A page tree comes back as markdown, not raw JSON."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/page.html", b"<html></html>")
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "heading",
                    "name": "Example Domain",
                    "children": [{"text": "Example Domain"}]
                },
                {
                    "role": "paragraph",
                    "children": [{"text": "This is a test page for browser tests."}]
                }
            ],
        }
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = (json.dumps(tree), False)
            result = await run_text(read_media_or_doc_tool, "upload/page.html")

        assert "# Example Domain" in result
        assert "This is a test page for browser tests." in result
        assert '"role"' not in result

    async def test_json_table_tree_converts_to_pipe_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A page tree with a table comes back as a pipe table, not raw JSON."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/page.html", b"<html></html>")
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "table",
                    "children": [
                        {
                            "role": "row",
                            "children": [
                                {"role": "cell", "children": [{"text": "Col A"}]},
                                {"role": "cell", "children": [{"text": "Col B"}]},
                            ],
                        },
                        {
                            "role": "row",
                            "children": [
                                {"role": "cell", "children": [{"text": "1"}]},
                                {"role": "cell", "children": [{"text": "2"}]},
                            ],
                        },
                    ],
                }
            ],
        }
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = (json.dumps(tree), False)
            result = await run_text(read_media_or_doc_tool, "upload/page.html")

        assert "| Col A | Col B |" in result
        assert "| --- | --- |" in result
        assert "| 1 | 2 |" in result
        assert '"role"' not in result

    async def test_raw_json_dict_passes_through_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSON with no role key comes back unchanged."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/data.json")
        raw_json = json.dumps(
            {"current_condition": [{"temp_C": "25", "weatherDesc": [{"value": "Sunny"}]}]}
        )
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = (raw_json, False)
            result = await run_text(read_media_or_doc_tool, "upload/data.json")

        assert result == raw_json

    async def test_plain_text_passes_through_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-JSON text content is returned unchanged."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/notes.txt")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = ("Plain text content", False)
            result = await run_text(read_media_or_doc_tool, "upload/notes.txt")

        assert result == "Plain text content"


class TestPrompt:
    """Tests for answering a question about the content with the LLM."""

    async def test_prompted_unprocessed_content_gets_llm_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt on unprocessed content comes back as the LLM answer."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("A lecture about AI.", False)
            mock_reason.return_value = ("About deep learning.", 0)
            result = await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")

        assert result == "About deep learning."
        mock_reason.assert_called_once_with("summarize", context="A lecture about AI.")

    async def test_prompt_reaches_resolver_as_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The prompt is handed to the resolver as its context."""
        _point_resources_at(tmp_path, monkeypatch)
        media = _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("A lecture about AI.", False)
            mock_reason.return_value = ("About deep learning.", 0)
            await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")

        mock_resolve.assert_called_once_with(str(media), context="summarize", verbose=True)

    async def test_no_prompt_never_calls_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no prompt, the raw content comes back and the LLM is not called."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("A lecture about AI.", False)
            result = await run_text(read_media_or_doc_tool, "upload/lecture.mp4")

        assert result == "A lecture about AI."
        mock_reason.assert_not_called()

    async def test_prompted_processed_content_comes_back_raw(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content that is already worked on comes back as is even with a prompt."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("Already processed content.", True)
            result = await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")

        assert result == "Already processed content."
        mock_reason.assert_not_called()

    async def test_llm_failure_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed answer gives back a failed tool message with the error text."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("A lecture about AI.", False)
            mock_reason.return_value = ("boom", 1)
            result = await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")

        assert result.startswith("Tool 'read_media_or_doc' failed: ")
        assert "Error extracting from 'upload/lecture.mp4': boom" in result


class TestMaxChar:
    """Tests for long results."""

    async def test_long_text_comes_back_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Text over the 25000 character default limit comes back whole, with no note."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/big.txt")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = (
                "x" * (DEFAULT_TOOL_OUTPUT_LEN_LIMIT + _OVERFLOW), False
            )
            result = await run_text(read_media_or_doc_tool, "upload/big.txt")

        assert result == "x" * (DEFAULT_TOOL_OUTPUT_LEN_LIMIT + _OVERFLOW)
        assert "[Truncated" not in result

    async def test_short_text_not_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Text under the limit is returned in full."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/small.txt")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = ("short content", False)
            result = await run_text(read_media_or_doc_tool, "upload/small.txt")

        assert result == "short content"


class TestCaching:
    """Tests for reusing a saved result."""

    async def test_identical_calls_reuse_stored_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two calls for the same file only read the file once."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = ("A lecture about AI.", False)
            first = await run_text(read_media_or_doc_tool, "upload/lecture.mp4")
            second = await run_text(read_media_or_doc_tool, "upload/lecture.mp4")

        assert first == "A lecture about AI."
        assert second == "A lecture about AI."
        assert mock_resolve.call_count == 1

    async def test_same_prompt_hits_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second call with the same prompt still reuses the saved result."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("A lecture about AI.", False)
            mock_reason.return_value = ("About deep learning.", 0)
            first = await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")
            second = await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")

        assert first == "About deep learning."
        assert second == "About deep learning."
        assert mock_resolve.call_count == 1

    async def test_different_prompt_misses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second call with a different prompt reads the file again."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/lecture.mp4")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve, patch(
            "app.lib.shared.format.reason"
        ) as mock_reason:
            mock_resolve.return_value = ("A lecture about AI.", False)
            mock_reason.return_value = ("About deep learning.", 0)
            await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="summarize")
            await run_text(read_media_or_doc_tool, "upload/lecture.mp4", prompt="what topic?")

        assert mock_resolve.call_count == 2

    async def test_changed_file_content_evicts_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that changes is read again."""
        _point_resources_at(tmp_path, monkeypatch)
        path = _share_file(tmp_path, "upload/lecture.mp4", b"AAAA")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = ("A lecture about AI.", False)
            await run_text(read_media_or_doc_tool, "upload/lecture.mp4")
            path.write_bytes(b"BBBB")
            await run_text(read_media_or_doc_tool, "upload/lecture.mp4")

        assert mock_resolve.call_count == 2

    async def test_failed_run_is_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that fails is not saved, so the same call runs again."""
        _point_resources_at(tmp_path, monkeypatch)
        _share_file(tmp_path, "upload/clip.mp3")
        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.side_effect = ResolveError("boom")
            first = await run_text(read_media_or_doc_tool, "upload/clip.mp3")

        assert first.startswith("Tool 'read_media_or_doc' failed: ")

        with patch("app.tools.read_media_or_doc.resolve_input") as mock_resolve:
            mock_resolve.return_value = ("fine", False)
            second = await run_text(read_media_or_doc_tool, "upload/clip.mp3")

        assert second == "fine"
        assert mock_resolve.call_count == 1
