"""Where the model files live on disk."""

__all__ = [
    "get_model_path",
    "get_models_dir",
]

import os


def get_models_dir() -> str:
    """Return the folder that holds the model files.

    Returns:
        The MODELS_DIR environment value, or the models folder in the project.
    """
    env_path = os.getenv("MODELS_DIR")
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")


def get_model_path(filename: str) -> str:
    """Return the full path to a model file by name."""
    return os.path.join(get_models_dir(), filename)
