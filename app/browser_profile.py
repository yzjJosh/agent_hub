"""Write the download defaults in the Chrome profile's Default/Preferences.

Chrome reads the file at startup, and resets the browser download path to
the values written there when a client disconnects.
"""

from __future__ import annotations

import json
import os

from app.lib.shared import paths

__all__ = []

_PROFILE_DIR = "/root/.config/google-chrome-ai"


def main() -> None:
    """Write the download defaults in the Chrome profile.

    The profile dir comes from the CHROME_PROFILE_DIR env var, and falls
    back to the profile the container's Chrome uses when it is unset or empty.
    """
    _set_download_defaults(os.environ.get("CHROME_PROFILE_DIR") or _PROFILE_DIR)


def _set_download_defaults(profile_dir: str) -> None:
    """Set the download defaults in the profile's Preferences file.

    The file is left alone when both download keys already hold the right
    values. Otherwise the download keys are written, and the other keys in
    the file are kept.

    Args:
        profile_dir: The Chrome user data dir holding the Default profile.
    """
    default_dir = os.path.join(profile_dir, "Default")
    prefs_path = os.path.join(default_dir, "Preferences")
    target_dir = paths.download_dir()
    data = _read_prefs(prefs_path)
    download = data.setdefault("download", {})
    if download.get("default_directory") == target_dir and download.get("prompt_for_download") is False:
        return
    download["default_directory"] = target_dir
    download["prompt_for_download"] = False
    os.makedirs(default_dir, exist_ok=True)
    _write_prefs(prefs_path, data)


def _read_prefs(path: str) -> dict:
    """Read the Preferences file, and return an empty one when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _write_prefs(path: str, data: dict) -> None:
    """Write the Preferences file, replacing any file already there."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
