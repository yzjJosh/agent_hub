"""Tests for app.tools.present_to_user."""

import inspect
import logging
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import app.lib.db.store as store_module
from app.lib.render.descriptor import build_descriptor
from app.tools.present_to_user import present_to_user_tool
from tests.app.conftest import thumb_calls
from tests.app.lib.tool.conftest import run_text

_BASE = "http://localhost:8315"
_PREFIX = "Presented to the user at "
_SINGLE_SUFFIX = " · link valid until the file is garbage-collected (30 days)"
_MULTI_SUFFIX = " · links valid until the files are garbage-collected (30 days)"
_TOKEN_RE = re.compile(r"^[0-9a-f]{16}$")


def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the share folders and the base URL at temp values, then give back the resources and data folders."""
    resources = tmp_path / "resources"
    data = tmp_path / "data"
    resources.mkdir()
    data.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("DAV_BASE_URL", _BASE)
    return resources, data


def _share_file(resources: Path, rel: str, data: bytes = b"fake bytes") -> Path:
    """Create a file in the resource share and return its absolute path."""
    path = resources / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _content_files(resources: Path) -> list[Path]:
    """The presentation files written into the share."""
    return sorted((resources / "presentations").iterdir())


def _stub_identify(monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    """Make the type detector always give back this label."""

    async def fake_identify(_data: bytes) -> str:
        return label

    monkeypatch.setattr("app.tools.present_to_user.identify", fake_identify)


@pytest.fixture
async def present_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Path, Path]]:
    """Point the resource folder, the data folder, and the record store at temp values."""
    resources, data = _env(tmp_path, monkeypatch)
    previous_task = store_module.task_store
    previous_tool = store_module.tool_store
    store_module.init_store(str(data / "records.db"))
    await store_module.task_store.open()
    yield resources, data
    await store_module.task_store.close()
    store_module.task_store = previous_task
    store_module.tool_store = previous_tool


class TestValidation:
    """Tests for bad inputs."""

    @pytest.mark.parametrize(
        "files, content",
        [(None, None), ([], None), (None, ""), ([], "")],
    )
    async def test_requires_files_or_content(
        self,
        files: list[str] | None,
        content: str | None,
    ) -> None:
        """With no files and no content, the tool gives back the required input error."""
        result = await run_text(present_to_user_tool, title="t", files=files, content=content)

        assert result == "Error: at least one of files or content is required"

    async def test_missing_file_points_to_upload_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path gives back upload help and presents nothing."""
        _resources, _data = _env(tmp_path, monkeypatch)
        calls = thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="t", files=["upload/nope.jpg"])

        assert "Resource not found: upload/nope.jpg" in result
        assert "send_or_receive_file" in result
        assert calls == []

    async def test_path_escaping_share_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path that points outside the share is turned down and shows nothing."""
        resources, _data = _env(tmp_path, monkeypatch)
        outside = resources.parent / "outside_present_to_user"
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("secret content")
        result = await run_text(
            present_to_user_tool,
            title="t", files=["../outside_present_to_user/secret.txt"]
        )

        assert "Resource not found" in result
        assert "secret content" not in result

    async def test_directory_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A folder is turned down with a not a file error."""
        resources, _data = _env(tmp_path, monkeypatch)
        (resources / "upload").mkdir()
        result = await run_text(present_to_user_tool, title="t", files=["upload"])

        assert "'upload' is a folder, not a file" in result

    async def test_missing_file_reported_before_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path is reported even when an earlier entry is a folder."""
        resources, _data = _env(tmp_path, monkeypatch)
        (resources / "upload").mkdir()
        result = await run_text(present_to_user_tool, title="t", files=["upload", "upload/missing.png"])

        assert "Resource not found: upload/missing.png" in result

    async def test_invalid_files_element_returns_canonical_error(self) -> None:
        """A files item that is not text is rejected before anything is shown."""
        result = await run_text(present_to_user_tool, title="t", files=["upload/ok.png", 5])
        assert result == "Error: files.1: Input should be a valid string."


class TestPresentation:
    """Tests for the view links the tool gives back."""

    async def test_single_file_single_url_writes_row(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One file with no content gives a plain view link and writes one presentation row."""
        resources, data = present_env
        _share_file(resources, "upload/pic.png", b"\x89PNG fake png bytes")
        calls = thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="t", files=["upload/pic.png"])

        assert result == f"{_PREFIX}{_BASE}/view/upload/pic.png{_SINGLE_SUFFIX}"
        assert [call[2] for call in calls] == ["upload/pic.png"]
        assert all(call[0] for call in calls)
        assert not any(p.is_dir() for p in data.iterdir())
        assert not (resources / "presentations").exists()
        rows = await store_module.presentation_store.list_newest()
        assert len(rows) == 1
        assert rows[0].files == ["upload/pic.png"]
        assert rows[0].url_path == "/view/upload/pic.png"

    async def test_single_file_with_space_is_percent_encoded(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file name with a space gives an escaped view link."""
        resources, data = present_env
        _share_file(resources, "upload/my file.png", b"\x89PNG fake png bytes")
        calls = thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="t", files=["upload/my file.png"])

        assert result == f"{_PREFIX}{_BASE}/view/upload/my%20file.png{_SINGLE_SUFFIX}"
        assert [call[2] for call in calls] == ["upload/my file.png"]
        assert not any(p.is_dir() for p in data.iterdir())
        rows = await store_module.presentation_store.list_newest()
        assert [row.url_path for row in rows] == ["/view/upload/my%20file.png"]

    async def test_single_file_keeps_result_when_row_write_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A solo presentation gives back the same link and logs a warning when the row write fails."""
        resources, _data = _env(tmp_path, monkeypatch)
        _share_file(resources, "upload/pic.png", b"\x89PNG fake png bytes")
        calls = thumb_calls(monkeypatch)

        async def _raise(_id: str, _presented_at: str, _url_path: str, _files: list[str], **_kw: str | None) -> None:
            raise RuntimeError("row write failed")

        monkeypatch.setattr(store_module.presentation_store, "insert", _raise)
        with caplog.at_level(logging.WARNING, logger="app.tools.present_to_user"):
            result = await run_text(present_to_user_tool, title="t", files=["upload/pic.png"])

        assert result == f"{_PREFIX}{_BASE}/view/upload/pic.png{_SINGLE_SUFFIX}"
        assert [call[2] for call in calls] == ["upload/pic.png"]
        assert any("failed to write presentation row" in message for message in caplog.messages)

    async def test_multiple_files_fail_when_row_write_fails(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-file presentation gives back the failure result when the row write fails."""
        resources, _data = present_env
        _share_file(resources, "upload/a.png", b"aaa")
        _share_file(resources, "upload/b.txt", b"bbb")
        thumb_calls(monkeypatch)

        async def _raise(_id: str, _presented_at: str, _url_path: str, _files: list[str], **_kw: str | None) -> None:
            raise RuntimeError("row write failed")

        monkeypatch.setattr(store_module.presentation_store, "insert", _raise)
        result = await run_text(present_to_user_tool, title="t", files=["upload/a.png", "upload/b.txt"])

        assert result == "Tool 'present_to_user' failed: row write failed"
        assert await store_module.presentation_store.list_newest() == []

    async def test_content_only_materializes_share_file(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content with no files is written to a presentation file named after the detected type."""
        resources, data = present_env
        calls = thumb_calls(monkeypatch)
        _stub_identify(monkeypatch, "markdown")
        result = await run_text(present_to_user_tool, title="t", content="# hello world")

        files = _content_files(resources)
        assert len(files) == 1
        token = files[0].stem
        assert _TOKEN_RE.match(token)
        assert files[0].read_text(encoding="utf-8") == "# hello world"
        assert result == f"{_PREFIX}{_BASE}/view/presentations/{token}.md{_SINGLE_SUFFIX}"
        assert [call[2] for call in calls] == [f"presentations/{token}.md"]
        assert not any(p.is_dir() for p in data.iterdir())
        row = await store_module.presentation_store.get(token)
        assert row is not None and row.files == [f"presentations/{token}.md"]
        assert row.url_path == f"/view/presentations/{token}.md"

    async def test_content_and_file_creates_presentation_row(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content plus a file gives one link for both, with a presentation row listing them."""
        resources, _data = present_env
        _share_file(resources, "upload/pic.png", b"\x89PNG fake png bytes")
        calls = thumb_calls(monkeypatch)
        _stub_identify(monkeypatch, "txt")
        result = await run_text(present_to_user_tool, title="t", files=["upload/pic.png"], content="notes")

        files = _content_files(resources)
        assert len(files) == 1
        token = files[0].stem
        assert _TOKEN_RE.match(token)
        assert files[0].suffix == ".txt"
        assert files[0].read_text(encoding="utf-8") == "notes"
        assert result == (
            f"{_PREFIX}{_BASE}/view/presentation/{token} · 2 file(s){_MULTI_SUFFIX}"
        )
        row = await store_module.presentation_store.get(token)
        assert row is not None and row.id == token
        assert sorted(row.files) == sorted(
            ["upload/pic.png", f"presentations/{token}.txt"]
        )
        assert row.url_path == f"/view/presentation/{token}"
        assert {call[2] for call in calls} == {
            "upload/pic.png",
            f"presentations/{token}.txt",
        }

    async def test_multiple_files_no_content(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Many files with no content give one presentation link that counts the files."""
        resources, _data = present_env
        _share_file(resources, "upload/a.png", b"aaa")
        _share_file(resources, "upload/b.txt", b"bbb")
        _share_file(resources, "upload/c.md", b"ccc")
        calls = thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="t", files=["upload/a.png", "upload/b.txt", "upload/c.md"])

        rows = await store_module.presentation_store.list_newest()
        assert len(rows) == 1
        row = rows[0]
        rid = row.id
        assert _TOKEN_RE.match(rid)
        assert result == (
            f"{_PREFIX}{_BASE}/view/presentation/{rid} · 3 file(s){_MULTI_SUFFIX}"
        )
        assert row.files == ["upload/a.png", "upload/b.txt", "upload/c.md"]
        assert row.url_path == f"/view/presentation/{rid}"
        assert {call[2] for call in calls} == {
            "upload/a.png",
            "upload/b.txt",
            "upload/c.md",
        }
        assert not (resources / "presentations").exists()


_PYTHON_SOURCE = (
    'def greet(name):\n    return f"Hello, {name}"\n\nprint(greet("world"))\n'
)
_JSON_BODY = (
    '{\n  "name": "agent",\n  "version": "0.1.0",\n'
    '  "deps": ["fastmcp", "httpx"]\n}\n'
)
_MARKDOWN_BODY = "# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n"
_PROSE_BODY = (
    "The weather in the city was mild on Tuesday, and many people walked "
    "along the river after work. Shops stayed open late, and the park "
    "was full of families."
)


class TestTitle:
    """Tests for the title the tool stores on each presentation row."""

    async def test_title_stored_in_solo_row(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A solo presentation stores the title on its row and the link is unchanged."""
        resources, _data = present_env
        _share_file(resources, "upload/pic.png", b"\x89PNG fake png bytes")
        thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="My picture", files=["upload/pic.png"])

        assert result == f"{_PREFIX}{_BASE}/view/upload/pic.png{_SINGLE_SUFFIX}"
        rows = await store_module.presentation_store.list_newest()
        assert len(rows) == 1
        assert rows[0].title == "My picture"

    async def test_title_stored_in_multi_row(
        self, present_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-file presentation stores the title on its row."""
        resources, _data = present_env
        _share_file(resources, "upload/a.png", b"aaa")
        _share_file(resources, "upload/b.txt", b"bbb")
        thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="Two files", files=["upload/a.png", "upload/b.txt"])

        rows = await store_module.presentation_store.list_newest()
        assert len(rows) == 1
        assert rows[0].title == "Two files"
        assert result == (
            f"{_PREFIX}{_BASE}/view/presentation/{rows[0].id} · 2 file(s){_MULTI_SUFFIX}"
        )

    def test_missing_title_returns_canonical_required_error(self) -> None:
        """A call without the title gets back the framework's required param error."""
        error = present_to_user_tool.validate_params({"files": ["upload/pic.png"]})
        assert error == "Error: title: Missing required argument."


class TestContentExtension:
    """Tests for the extension a presentation file gets."""

    @pytest.mark.parametrize(
        ("label", "ext"),
        [
            ("python", "py"),
            ("javascript", "js"),
            ("typescript", "ts"),
            ("shell", "sh"),
            ("sql", "sql"),
            ("go", "go"),
            ("rust", "rs"),
            ("ruby", "rb"),
            ("c", "c"),
            ("cpp", "cpp"),
            ("cs", "cs"),
            ("java", "java"),
            ("php", "php"),
            ("toml", "toml"),
            ("kotlin", "kt"),
            ("lua", "lua"),
            ("swift", "swift"),
            ("scala", "scala"),
            ("r", "r"),
            ("haskell", "hs"),
            ("erlang", "erl"),
            ("json", "json"),
            ("jsonl", "jsonl"),
            ("csv", "csv"),
            ("yaml", "yaml"),
            ("xml", "xml"),
            ("html", "html"),
            ("markdown", "md"),
            ("txt", "txt"),
        ],
    )
    async def test_detected_label_materializes_its_extension(
        self,
        present_env: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        ext: str,
    ) -> None:
        """Each known type gives the presentation file that type's extension."""
        resources, _data = present_env
        thumb_calls(monkeypatch)
        _stub_identify(monkeypatch, label)
        await run_text(present_to_user_tool, title="t", content="sample body")

        files = _content_files(resources)
        assert len(files) == 1
        assert files[0].suffix == f".{ext}"

    @pytest.mark.parametrize("label", ["unknown", "empty", "randombytes", "pdf", "png"])
    async def test_unmapped_label_falls_back_to_markdown(
        self,
        present_env: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
        label: str,
    ) -> None:
        """A type with no extension of its own gives a .md presentation file."""
        resources, data = present_env
        thumb_calls(monkeypatch)
        _stub_identify(monkeypatch, label)
        result = await run_text(present_to_user_tool, title="t", content="sample body")

        files = _content_files(resources)
        assert len(files) == 1
        token = files[0].stem
        assert files[0].suffix == ".md"
        assert result == f"{_PREFIX}{_BASE}/view/presentations/{token}.md{_SINGLE_SUFFIX}"
        assert not any(p.is_dir() for p in data.iterdir())


class TestDetectedExtensionLive:
    """Live tests for the extension a presentation file gets."""

    @pytest.mark.live
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        ("content", "ext", "kind"),
        [
            (_PYTHON_SOURCE, ".py", "code"),
            (_JSON_BODY, ".json", "json"),
            (_MARKDOWN_BODY, ".md", "markdown"),
            (_PROSE_BODY, ".txt", "text"),
        ],
    )
    async def test_content_materializes_the_detected_extension(
        self,
        present_env: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
        content: str,
        ext: str,
        kind: str,
    ) -> None:
        """Each kind of text gets the right extension and the right preview kind."""
        resources, _data = present_env
        thumb_calls(monkeypatch)
        result = await run_text(present_to_user_tool, title="t", content=content)

        files = _content_files(resources)
        assert len(files) == 1
        file = files[0]
        token = file.stem
        assert file.suffix == ext
        assert file.read_text(encoding="utf-8") == content
        assert result == f"{_PREFIX}{_BASE}/view/presentations/{token}{ext}{_SINGLE_SUFFIX}"
        descriptor = await build_descriptor(f"presentations/{token}{ext}", str(file))
        assert descriptor["kind"] == kind


class TestUntracked:
    """Tests that the tool leaves no run records."""

    async def test_successful_call_creates_no_run_records(
        self,
        present_env: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A good call writes no task rows and no run folder, but writes one presentation row."""
        resources, _data = present_env
        _share_file(resources, "upload/pic.png", b"\x89PNG fake png bytes")
        _stub_identify(monkeypatch, "txt")
        result = await run_text(present_to_user_tool, title="t", files=["upload/pic.png"], content="notes")

        assert "presentation" in result
        stats = await store_module.tool_stats()
        assert [s for s in stats if s["tool_name"] == "present_to_user"] == []
        assert await store_module.total_count() == 0
        runs = os.path.join(store_module.tool_store.data_dir, "runs")
        entries = os.listdir(runs) if os.path.isdir(runs) else []
        assert entries == []
        rows = await store_module.presentation_store.list_newest()
        assert len(rows) == 1


class TestNoMicrobin:
    """Tests that the tool source has no pastebin code."""

    async def test_no_microbin_references_remain(self) -> None:
        """The tool source has no pastebin address and no code for uploading or sniffing file types."""
        assert await run_text(present_to_user_tool, title="t") == "Error: at least one of files or content is required"
        src = Path(inspect.getsourcefile(type(present_to_user_tool))).read_text(encoding="utf-8")
        assert "paste." not in src
        assert "microbin" not in src.lower()
        assert "httpx" not in src
        assert "magika" not in src.lower()


class TestAtLeastOneRuleOnParams:
    """Tests for where the at least one rule is written."""

    def test_rule_on_both_param_descriptions(self) -> None:
        rule = "At least one of files or content is required."
        descs = {p.name: p.description for p in present_to_user_tool.params}
        assert rule in descs["files"]
        assert rule in descs["content"]

    def test_rule_not_in_tool_description(self) -> None:
        assert "At least one of" not in present_to_user_tool.description

    def test_content_description_states_detected_type_materialization(self) -> None:
        """The content description says the detected type, not a markdown file."""
        descs = {p.name: p.description for p in present_to_user_tool.params}
        assert "markdown file" not in descs["content"].lower()
        assert "detected type" in descs["content"]
