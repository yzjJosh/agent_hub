"""Tests for the console preview script.

The tests run the script with a fake docker command that stands in for the
Agent Hub container, so no real preview server is ever started. They are skipped
when docker or git is missing.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "console_preview"
_UP_RE = re.compile(r"preview up on :(\d+) \(pid (\d+)\)")


def _host_prereqs_available() -> bool:
    """Check the host has everything the preview script needs."""
    return shutil.which("docker") is not None and shutil.which("git") is not None


skip_if = pytest.mark.skipif(
    not _host_prereqs_available(),
    reason="console_preview host environment missing, needs docker, git",
)

pytestmark: list[pytest.MarkDecorator] = [pytest.mark.live, skip_if]

_FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, re, sys

state = os.environ["FAKE_CONTAINER_STATE"]
ctmp = os.path.join(state, "tmp")
procs_path = os.path.join(state, "procs.json")

args = sys.argv[1:]
assert args[0] == "exec", args
i = 1
detached = False
eflags = []
while args[i].startswith("-"):
    if args[i] == "-w":
        i += 2
    elif args[i] == "-e":
        eflags.append(args[i + 1])
        i += 2
    elif args[i] == "-d":
        detached = True
        i += 1
    else:
        i += 1
container, *cmd = args[i:]
assert container == "agent_hub", args
last = cmd[-1]


def _procs():
    return json.load(open(procs_path)) if os.path.exists(procs_path) else {}


def _save(procs):
    json.dump(procs, open(procs_path, "w"))


def _in_container(path):
    return os.path.join(ctmp, path.removeprefix("/tmp/"))


if cmd[0] == "python" and "secrets" in last:
    print(os.environ.get("FAKE_PICK_PORT", "15000"))
elif cmd[0] == "python" and "create_connection" in last:
    port = re.search(r"\('127\.0\.0\.1', (\d+)\)", last).group(1)
    alive = {v["port"] for v in _procs().values()}
    sys.exit(0 if port in alive else 1)
elif detached and cmd[0] == "bash":
    env = {e.split("=", 1)[0]: e.split("=", 1)[1] for e in eflags}
    pid = str(4001 + len(_procs()))
    pid_file = re.search(r"echo \$\$ > (\S+);", last).group(1)
    marker = re.search(r"app\.main '([^']+)'", last).group(1)
    os.makedirs(os.path.dirname(_in_container(pid_file)), exist_ok=True)
    with open(_in_container(pid_file), "w") as fh:
        fh.write(pid + "\n")
    procs = _procs()
    procs[pid] = {"port": env["HTTP_PORT"], "marker": marker}
    _save(procs)
    record = {"HTTP_PORT": env["HTTP_PORT"], "DATA_DIR": env["DATA_DIR"],
              "RESOURCES_DIR": env["RESOURCES_DIR"], "marker": marker}
    with open(os.path.join(state, "launches.log"), "a") as fh:
        fh.write(json.dumps(record) + "\n")
elif cmd[0] == "cat":
    path = _in_container(cmd[1])
    if not os.path.exists(path):
        sys.exit(1)
    print(open(path).read().strip())
elif cmd[0] == "bash" and "/proc/" in last:
    pid = re.search(r"/proc/(\d+)/cmdline", last).group(1)
    marker = re.search(r"grep -qF '([^']+)'", last).group(1)
    sys.exit(0 if _procs().get(pid, {}).get("marker") == marker else 1)
elif cmd[0] == "bash" and last.startswith("kill "):
    pid = last.split()[1]
    procs = _procs()
    procs.pop(pid, None)
    _save(procs)
    with open(os.path.join(state, "killed.log"), "a") as fh:
        fh.write(pid + "\n")
else:
    raise AssertionError(f"unexpected container command: {cmd!r}")
'''

_GIT_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.com",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.com",
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), env=_GIT_ENV, check=True, capture_output=True)


@pytest.fixture
def worktrees(tmp_path: Path) -> dict[str, Path]:
    """A base repo plus two worktrees named wA and wB."""
    base = tmp_path / "base"
    base.mkdir()
    _git("init", cwd=base)
    (base / "f.txt").write_text("x")
    _git("add", "f.txt", cwd=base)
    _git("commit", "-m", "base", cwd=base)
    out = {}
    for name in ("wA", "wB"):
        wt = tmp_path / name
        _git("worktree", "add", str(wt), "-b", name, cwd=base)
        out[name] = wt
    return out


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    """A folder holding a fake docker command that stands in for the Agent Hub container."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    docker = fakebin / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    os.chmod(docker, 0o755)
    return fakebin


def _preview(cwd: Path, env: dict, fakebin: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    full_env = dict(os.environ)
    full_env["PATH"] = f"{fakebin}{os.pathsep}{full_env.get('PATH', '')}"
    full_env.update(env)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        env=full_env,
    )


def _state_env(store: Path, state: Path) -> dict[str, str]:
    state.mkdir(parents=True, exist_ok=True)
    return {"PREVIEW_STORE_ROOT": str(store), "FAKE_CONTAINER_STATE": str(state)}


def _start(worktree: Path, env: dict, fakebin: Path) -> tuple[str, str]:
    result = _preview(worktree, env, fakebin, "start")
    assert result.returncode == 0, result.stderr
    match = _UP_RE.search(result.stdout) or re.search(r"preview already running on :(\d+) \(pid (\d+)\)", result.stdout)
    assert match, result.stdout
    return match.group(1), match.group(2)


def _kill_emulated(state: Path, pid: str) -> None:
    """Make the preview process look like it died on its own."""
    procs_path = state / "procs.json"
    procs = json.loads(procs_path.read_text())
    procs.pop(pid, None)
    procs_path.write_text(json.dumps(procs))


class TestStart:
    def test_start_prints_port_and_pid_and_writes_store(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        port, pid = _start(worktrees["wA"], _state_env(store, state), fake_docker)
        assert port == "15000"  # the port the fake picks
        assert (store / "wA" / "port").read_text().strip() == port
        assert (store / "wA" / "pid").read_text().strip() == pid
        launches = [json.loads(line) for line in (state / "launches.log").read_text().splitlines()]
        assert len(launches) == 1
        assert launches[0]["DATA_DIR"] == "/tmp/console-preview-wA-data"
        assert launches[0]["RESOURCES_DIR"] == "/tmp/console-preview-wA-resources"
        assert launches[0]["marker"] == "agent-console-preview-wA"

    def test_start_consumes_container_pick(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        env["FAKE_PICK_PORT"] = "16666"
        port, _ = _start(worktrees["wA"], env, fake_docker)
        assert port == "16666"  # the script uses the port the container picks

    def test_start_twice_reports_already_running_and_launches_once(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        port, pid = _start(worktrees["wA"], env, fake_docker)
        again = _preview(worktrees["wA"], env, fake_docker, "start")
        assert again.returncode == 0
        assert f"preview already running on :{port} (pid {pid})" in again.stdout
        assert len((state / "launches.log").read_text().splitlines()) == 1

    def test_start_after_crash_recovers(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        _, pid = _start(worktrees["wA"], env, fake_docker)
        _kill_emulated(state, pid)
        _start(worktrees["wA"], env, fake_docker)  # must not report already-running
        assert len((state / "launches.log").read_text().splitlines()) == 2


class TestStatus:
    def test_status_up_reports_stored_port(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        port, pid = _start(worktrees["wA"], env, fake_docker)
        result = _preview(worktrees["wA"], env, fake_docker, "status")
        assert result.returncode == 0
        assert f"preview up on :{port} (pid {pid})" in result.stdout

    def test_status_without_store_reports_not_running(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        result = _preview(worktrees["wA"], env, fake_docker, "status")
        assert result.returncode == 1
        assert "preview not running" in result.stdout

    def test_status_after_death_reports_not_running(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        port, pid = _start(worktrees["wA"], env, fake_docker)
        _kill_emulated(state, pid)
        result = _preview(worktrees["wA"], env, fake_docker, "status")
        assert result.returncode == 1
        assert f"preview not running (stored port :{port} not answering)" in result.stdout


class TestStop:
    def test_stop_kills_only_stored_pid_and_removes_store(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        port, pid = _start(worktrees["wA"], env, fake_docker)
        result = _preview(worktrees["wA"], env, fake_docker, "stop")
        assert result.returncode == 0
        assert f"preview stopped (pid {pid}, was on :{port})" in result.stdout
        assert (state / "killed.log").read_text().split() == [pid]
        assert not (store / "wA").exists()

    def test_stop_without_store_is_noop(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        result = _preview(worktrees["wA"], env, fake_docker, "stop")
        assert result.returncode == 0
        assert "preview not running" in result.stdout

    def test_stop_after_death_reports_not_running_and_removes_store(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        _, pid = _start(worktrees["wA"], env, fake_docker)
        _kill_emulated(state, pid)
        result = _preview(worktrees["wA"], env, fake_docker, "stop")
        assert result.returncode == 0
        assert f"preview not running (stored pid {pid} no longer matches the marker)" in result.stdout
        assert not (store / "wA").exists()
        assert not (state / "killed.log").exists()  # nothing was killed


class TestMultiWorktreeIsolation:
    def test_stop_wa_leaves_wb_running(self, worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path) -> None:
        # Both worktrees share one container and one store root.
        store, state = tmp_path / "store", tmp_path / "cstate"
        env = _state_env(store, state)
        _, pid_a = _start(worktrees["wA"], env, fake_docker)
        port_b, pid_b = _start(worktrees["wB"], env, fake_docker)
        assert pid_a != pid_b
        stop_a = _preview(worktrees["wA"], env, fake_docker, "stop")
        assert stop_a.returncode == 0
        assert "preview stopped" in stop_a.stdout
        assert (state / "killed.log").read_text().split() == [pid_a]  # only A's PID
        status_b = _preview(worktrees["wB"], env, fake_docker, "status")
        assert status_b.returncode == 0
        assert f"preview up on :{port_b} (pid {pid_b})" in status_b.stdout
        launches = [json.loads(line) for line in (state / "launches.log").read_text().splitlines()]
        assert launches[1]["DATA_DIR"] == "/tmp/console-preview-wB-data"


def test_preview_store_root_is_documented_and_used(worktrees: dict[str, Path], fake_docker: Path, tmp_path: Path,
                                                   host_script_vars: set[str]) -> None:
    """PREVIEW_STORE_ROOT has an entry in .env.example part 3, and console_preview keeps its files there."""
    assert "PREVIEW_STORE_ROOT" in host_script_vars
    store = tmp_path / "store"
    env = _state_env(store, tmp_path / "cstate")
    port, _ = _start(worktrees["wA"], env, fake_docker)
    assert (store / "wA" / "port").read_text().strip() == port
    assert _preview(worktrees["wA"], env, fake_docker, "stop").returncode == 0
