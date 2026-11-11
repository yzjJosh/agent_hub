"""Works out what kind of text a stored body holds."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import onnxruntime as ort

from app.lib.shared.model_paths import get_model_path
from app.lib.render.classify import RenderKind

try:
    from magika import Magika
except Exception:
    Magika = None

__all__ = ["identify", "kind_for_label", "language_for_label"]

_UNKNOWN = "unknown"
_MODEL_REL_PATH = "magika/standard_v3_3"
_CPU_PROVIDER = "CPUExecutionProvider"
# Providers that run the model on an accelerator.
_ACCELERATOR_PROVIDERS = frozenset({
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "ROCMExecutionProvider",
})
_session: Magika | None = None
_session_lock = threading.Lock()

# Labels that have a render kind of their own.
_KIND_BY_LABEL: dict[str, RenderKind] = {
    "markdown": RenderKind.MARKDOWN,
    "json": RenderKind.JSON,
    "jsonl": RenderKind.JSONL,
    "csv": RenderKind.CSV,
    "yaml": RenderKind.YAML,
    "xml": RenderKind.XML,
    "html": RenderKind.HTML,
    "txt": RenderKind.TEXT,
}

# Code labels and the language name each one is highlighted with.
_LANGUAGE_BY_LABEL: dict[str, str] = {
    "c": "c",
    "cpp": "cpp",
    "cs": "csharp",
    "erlang": "erlang",
    "go": "go",
    "haskell": "haskell",
    "java": "java",
    "javascript": "javascript",
    "kotlin": "kotlin",
    "lua": "lua",
    "php": "php",
    "python": "python",
    "r": "r",
    "ruby": "ruby",
    "rust": "rust",
    "scala": "scala",
    "shell": "sh",
    "sql": "sql",
    "swift": "swift",
    "toml": "toml",
    "typescript": "typescript",
}


def kind_for_label(label: str) -> RenderKind:
    """Map a content label to its render kind.

    Args:
        label: A content type label.

    Returns:
        The dedicated kind for text-format labels, CODE for code labels,
        TEXT for every other label.
    """
    kind = _KIND_BY_LABEL.get(label)
    if kind is not None:
        return kind
    if label in _LANGUAGE_BY_LABEL:
        return RenderKind.CODE
    return RenderKind.TEXT


def language_for_label(label: str) -> str | None:
    """Map a content label to the language it is highlighted with.

    Args:
        label: A content type label.

    Returns:
        The language name for code labels, None for every other label.
    """
    return _LANGUAGE_BY_LABEL.get(label)


def _build_session() -> Magika:
    """Create the shared model session.

    Returns:
        The session the model runs on.

    Raises:
        RuntimeError: If the model cannot be set up to run on the CPU alone.
    """
    if Magika is None:
        raise RuntimeError("magika package is not importable")
    active = set(ort.get_available_providers()) & _ACCELERATOR_PROVIDERS
    if active:
        raise RuntimeError(f"accelerator providers active {sorted(active)}")
    model_dir = Path(get_model_path(_MODEL_REL_PATH))
    # The library does not hand out the session it makes. This check builds its own.
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=[_CPU_PROVIDER]
    )
    providers = session.get_providers()
    if any(provider != _CPU_PROVIDER for provider in providers):
        raise RuntimeError(f"non-CPU-only providers {providers}")
    return Magika(model_dir=model_dir)


def _get_session() -> Magika:
    """The shared model session, made the first time it is asked for."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = _build_session()
    return _session


def _identify_sync(data: bytes | None) -> str:
    if not data:
        return _UNKNOWN
    try:
        return _get_session().identify_bytes(data).output.label
    except Exception as exc:
        print(
            f"[text_type.detector._identify_sync] detection failed: {exc}",
            file=sys.stderr,
        )
        return _UNKNOWN


async def identify(data: bytes | None) -> str:
    """Work out what kind of text some bytes hold.

    Args:
        data: The raw bytes to look at.

    Returns:
        The label for the text, or 'unknown' when it cannot be worked out.
    """
    return await asyncio.to_thread(_identify_sync, data)
