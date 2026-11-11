import json
from pathlib import Path

import pytest

from app.browser_profile import main

_DOWNLOAD_DIR = "/dev/shm/agent/downloads"
_SHM_DOWNLOAD_DIR = "/dev/shm/downloads"
_MINIMAL_PREFERENCES = {
    "download": {
        "default_directory": _DOWNLOAD_DIR,
        "prompt_for_download": False,
    }
}


@pytest.fixture(autouse=True)
def _default_scratch_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test with the default scratch root."""
    monkeypatch.delenv("SCRATCH_DIR", raising=False)


def _write_prefs(profile: Path, data: dict) -> Path:
    """Write a Preferences file under the profile, and give back its path."""
    default = profile / "Default"
    default.mkdir(parents=True, exist_ok=True)
    prefs = default / "Preferences"
    prefs.write_text(json.dumps(data), encoding="utf-8")
    return prefs


def _read_prefs(prefs: Path) -> dict:
    """Read a Preferences file and give back its contents."""
    return json.loads(prefs.read_text(encoding="utf-8"))


def _run(profile: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point main at a throwaway profile, run it, and give back the Preferences path."""
    monkeypatch.setenv("CHROME_PROFILE_DIR", str(profile))
    main()
    return profile / "Default" / "Preferences"


def test_creates_prefs_for_new_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile without a Default folder gets a minimal Preferences file."""
    prefs = _run(tmp_path, monkeypatch)
    assert _read_prefs(prefs) == _MINIMAL_PREFERENCES


def test_keeps_other_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys outside the download block, and keys inside it, stay as they are."""
    existing = _write_prefs(tmp_path, {"profile": {"name": "Default"},
                                        "download": {"directory_list": ["/root/Downloads"]}})
    prefs = _run(tmp_path, monkeypatch)
    assert prefs == existing
    data = _read_prefs(prefs)
    assert data["profile"] == {"name": "Default"}
    assert data["download"]["directory_list"] == ["/root/Downloads"]
    assert data["download"]["default_directory"] == _DOWNLOAD_DIR
    assert data["download"]["prompt_for_download"] is False


def test_adds_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download block with only one of the two keys gets the other added."""
    _write_prefs(tmp_path, {"download": {"default_directory": _DOWNLOAD_DIR}})
    prefs = _run(tmp_path, monkeypatch)
    assert _read_prefs(prefs)["download"] == {"default_directory": _DOWNLOAD_DIR,
                                              "prompt_for_download": False}


def test_replaces_wrong_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrong download values are set to the shared folder and the prompt off."""
    _write_prefs(tmp_path, {"download": {"default_directory": "/root/Downloads",
                                          "prompt_for_download": True}})
    prefs = _run(tmp_path, monkeypatch)
    assert _read_prefs(prefs)["download"] == {"default_directory": _DOWNLOAD_DIR,
                                              "prompt_for_download": False}


def test_noop_when_values_already_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that already holds the right values is not rewritten."""
    prefs = _write_prefs(tmp_path, {"download": {"default_directory": _DOWNLOAD_DIR,
                                                  "prompt_for_download": False,
                                                  "directory_list": ["/root/Downloads"]}})
    mtime = prefs.stat().st_mtime_ns
    _run(tmp_path, monkeypatch)
    assert prefs.stat().st_mtime_ns == mtime
    assert _read_prefs(prefs) == {"download": {"default_directory": _DOWNLOAD_DIR,
                                               "prompt_for_download": False,
                                               "directory_list": ["/root/Downloads"]}}


def test_replaces_shm_download_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored download folder of /dev/shm/downloads is replaced with the scratch downloads folder."""
    _write_prefs(tmp_path, {"download": {"default_directory": _SHM_DOWNLOAD_DIR,
                                          "prompt_for_download": False}})
    prefs = _run(tmp_path, monkeypatch)
    assert _read_prefs(prefs) == _MINIMAL_PREFERENCES


def test_uses_scratch_root_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With SCRATCH_DIR set, the download folder is the downloads folder under it."""
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH_DIR", str(scratch))
    prefs = _run(tmp_path / "profile", monkeypatch)
    assert _read_prefs(prefs)["download"] == {"default_directory": str(scratch / "downloads"),
                                              "prompt_for_download": False}


def test_recovers_from_unreadable_prefs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt Preferences file is replaced with the minimal one."""
    default = tmp_path / "Default"
    default.mkdir()
    prefs = default / "Preferences"
    prefs.write_text("not json", encoding="utf-8")
    prefs = _run(tmp_path, monkeypatch)
    assert _read_prefs(prefs) == _MINIMAL_PREFERENCES
