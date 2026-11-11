"""Tests for the text type detector."""

import importlib
from pathlib import Path

from unittest.mock import MagicMock

import onnxruntime as ort
import pytest

import app.lib.text_type.detector as detector
from app.lib.shared.model_paths import get_model_path
from app.lib.render.classify import RenderKind

_PYTHON_SOURCE = b'def greet(name):\n    return f"Hello, {name}"\n\nprint(greet("world"))\n'

_PROSE = (
    "The weather in the city was mild on Tuesday, and many people walked "
    "along the river after work. Shops stayed open late, and the park "
    "was full of families."
)


@pytest.fixture(autouse=True)
def reset_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with a fresh detector and leave no built session behind."""
    importlib.reload(detector)
    monkeypatch.setattr(detector, "_session", None)


def _stub_magika(monkeypatch: pytest.MonkeyPatch, instances: list) -> None:
    """Point the detector at a fake model that records each build."""

    def fake_magika(model_dir: Path) -> MagicMock:
        magika = MagicMock()
        magika.model_dir = model_dir
        magika.identify_bytes.return_value.output.label = "python"
        instances.append(magika)
        return magika

    monkeypatch.setattr(detector, "Magika", fake_magika)


def _stub_session(monkeypatch: pytest.MonkeyPatch, providers: list) -> None:
    """Point the detector at a fake model session with the given providers."""
    session = MagicMock()
    session.get_providers.return_value = providers

    def fake_session(*_args: str, **_kwargs: list[str]) -> MagicMock:
        return session

    monkeypatch.setattr(ort, "InferenceSession", fake_session)


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("markdown", RenderKind.MARKDOWN),
        ("json", RenderKind.JSON),
        ("jsonl", RenderKind.JSONL),
        ("csv", RenderKind.CSV),
        ("yaml", RenderKind.YAML),
        ("xml", RenderKind.XML),
        ("html", RenderKind.HTML),
        ("txt", RenderKind.TEXT),
        ("python", RenderKind.CODE),
        ("javascript", RenderKind.CODE),
        ("shell", RenderKind.CODE),
        ("sql", RenderKind.CODE),
        ("go", RenderKind.CODE),
        ("pdf", RenderKind.TEXT),
        ("png", RenderKind.TEXT),
        ("unknown", RenderKind.TEXT),
        ("empty", RenderKind.TEXT),
        ("randombytes", RenderKind.TEXT),
    ],
)
def test_kind_for_label(label: str, kind: RenderKind) -> None:
    """Each label maps to its render kind."""
    assert detector.kind_for_label(label) == kind


@pytest.mark.parametrize(
    ("label", "language"),
    [
        ("python", "python"),
        ("javascript", "javascript"),
        ("shell", "sh"),
        ("sql", "sql"),
        ("go", "go"),
        ("typescript", "typescript"),
        ("rust", "rust"),
        ("c", "c"),
        ("cpp", "cpp"),
        ("java", "java"),
        ("php", "php"),
        ("ruby", "ruby"),
        ("toml", "toml"),
        ("kotlin", "kotlin"),
        ("lua", "lua"),
        ("swift", "swift"),
        ("scala", "scala"),
        ("r", "r"),
        ("haskell", "haskell"),
        ("cs", "csharp"),
        ("erlang", "erlang"),
        ("markdown", None),
        ("json", None),
        ("txt", None),
        ("unknown", None),
    ],
)
def test_language_for_label(label: str, language: str | None) -> None:
    """Each code label maps to its language name."""
    assert detector.language_for_label(label) == language


async def test_identify_builds_the_session_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls give the same answer and build the model once."""
    instances: list = []
    _stub_magika(monkeypatch, instances)
    _stub_session(monkeypatch, ["CPUExecutionProvider"])
    assert await detector.identify(_PYTHON_SOURCE) == "python"
    assert await detector.identify(_PYTHON_SOURCE) == "python"
    assert len(instances) == 1


async def test_identify_returns_unknown_when_the_model_dir_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing model folder gives back unknown."""
    instances: list = []
    _stub_magika(monkeypatch, instances)
    monkeypatch.setattr(detector, "get_model_path", lambda _name: "/no/such/models")
    assert await detector.identify(b"hi") == "unknown"
    assert len(instances) == 0


async def test_identify_returns_unknown_when_the_session_is_not_cpu_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model session with an accelerator gives back unknown."""
    instances: list = []
    _stub_magika(monkeypatch, instances)
    _stub_session(monkeypatch, ["CUDAExecutionProvider"])
    assert await detector.identify(b"print(1)") == "unknown"
    assert len(instances) == 0


async def test_identify_returns_unknown_when_an_accelerator_provider_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accelerator on the machine gives back unknown."""
    instances: list = []
    _stub_magika(monkeypatch, instances)
    _stub_session(monkeypatch, ["CPUExecutionProvider"])
    monkeypatch.setattr(
        ort, "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    assert await detector.identify(b"print(1)") == "unknown"
    assert len(instances) == 0


async def test_identify_builds_when_no_accelerator_provider_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other providers on the machine still let the model build."""
    instances: list = []
    _stub_magika(monkeypatch, instances)
    _stub_session(monkeypatch, ["CPUExecutionProvider"])
    monkeypatch.setattr(
        ort, "get_available_providers",
        lambda: ["AzureExecutionProvider", "CPUExecutionProvider"],
    )
    assert await detector.identify(b"print(1)") == "python"
    assert len(instances) == 1


@pytest.mark.parametrize("data", [b"", None])
async def test_identify_returns_unknown_for_empty_input(data: bytes | None) -> None:
    """Empty input gives back unknown."""
    assert await detector.identify(data) == "unknown"


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("data", "label"),
    [
        (_PYTHON_SOURCE, "python"),
        (
            b'{\n  "name": "agent",\n  "version": "0.1.0",\n  "deps": ["fastmcp", "httpx"]\n}\n',
            "json",
        ),
        (b"name: agent\nversion: 0.1.0\nitems:\n  - one\n  - two\n", "yaml"),
        (b"name,amount\nalpha,1\nbeta,2\ngamma,3\n", "csv"),
        (
            b"# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n",
            "markdown",
        ),
        (
            b"<!doctype html>\n<html>\n<head><title>demo</title></head>"
            b"\n<body><p>hello</p></body>\n</html>\n",
            "html",
        ),
        (
            b"CREATE TABLE items (name TEXT, amount INT);\n"
            b"SELECT name FROM items WHERE amount > 1;\n",
            "sql",
        ),
        (b'#!/bin/sh\necho "starting"\nsleep 1\nexit 0\n', "shell"),
        (_PROSE.encode("utf-8"), "txt"),
    ],
)
async def test_live_identify_matrix(data: bytes, label: str) -> None:
    """The real model identifies each text type."""
    assert await detector.identify(data) == label


@pytest.mark.live
@pytest.mark.timeout(120)
async def test_live_session_runs_cpu_only() -> None:
    """The real model runs on the CPU only."""
    model_path = Path(get_model_path("magika/standard_v3_3")) / "model.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert await detector.identify(_PYTHON_SOURCE) == "python"
