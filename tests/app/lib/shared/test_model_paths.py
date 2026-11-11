"""Tests for app.lib.shared.model_paths module."""

import os
from typing import Self

import pytest

from app.lib.shared.model_paths import get_model_path, get_models_dir


class TestGetModelsDir:
    """Test the get_models_dir function."""

    def test_returns_env_var_when_set(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A models folder that is set is returned as is."""
        monkeypatch.setenv("MODELS_DIR", "/custom/models")
        assert get_models_dir() == "/custom/models"

    def test_returns_default_path(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing set, the models folder is a full path with models in it."""
        monkeypatch.delenv("MODELS_DIR", raising=False)
        result = get_models_dir()
        assert "models" in result
        assert os.path.isabs(result)

    def test_empty_env_var_falls_back_to_default(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty setting falls back to the default models folder."""
        monkeypatch.setenv("MODELS_DIR", "")
        result = get_models_dir()
        assert "models" in result


class TestGetModelPath:
    """Test the get_model_path function."""

    def test_returns_joined_path(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model file name is joined onto the models folder."""
        monkeypatch.setenv("MODELS_DIR", "/models")
        result = get_model_path("test.onnx")
        assert result == "/models/test.onnx"

    def test_handles_nested_filename(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A name with a folder in it is kept whole."""
        monkeypatch.setenv("MODELS_DIR", "/models")
        result = get_model_path("sub/model.onnx")
        assert "sub/model.onnx" in result
