"""Tests for app.lib.shared.config module."""

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from app.lib.shared import config
from tests import ENV_EXAMPLE, EnvExample, read_env_example

_BIND_VARS = ("HTTP_HOST", "HTTP_PORT", "SSE_HOST", "SSE_PORT")
_REASON_VARS = (
    "REASON_BACKEND_URL", "REASON_MODEL", "REASON_API_KEY",
    "REASON_THINKING_BACKEND_URL", "REASON_THINKING_MODEL", "REASON_THINKING_API_KEY",
)

_OTHER_PROGRAM_VARS = {"TZ", "GITHUB_TOKEN", "AUTH_TOKEN", "CT0"}
_ENV_GETTERS = {"os.getenv", "os.environ.get", "require_env", "config.require_env"}
_CONFIG_SOURCE = Path("app", "lib", "shared", "config.py")
_PROFILES = (config.REASON, config.REASON_THINKING, config.OCR, config.EMBEDDING, config.RERANKER, config.TTS)


@pytest.fixture
def _clean_bind_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with no bind variable set."""
    for name in _BIND_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _clean_reason_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with no reason backend variable set."""
    for name in _REASON_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def env_example() -> EnvExample:
    """The entries of the repository's .env.example, by part."""
    if not ENV_EXAMPLE.is_file():
        pytest.skip(f"{ENV_EXAMPLE} is not mounted in this container")
    return read_env_example()


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Set each variable in env."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, ("0.0.0.0", 80)),
        ({"HTTP_HOST": "10.0.0.5", "HTTP_PORT": "8080"}, ("10.0.0.5", 8080)),
        (
            {"HTTP_HOST": "10.0.0.5", "HTTP_PORT": "8080", "SSE_HOST": "10.0.0.9", "SSE_PORT": "9000"},
            ("10.0.0.5", 8080),
        ),
        ({"SSE_HOST": "10.0.0.9", "SSE_PORT": "9000"}, ("10.0.0.9", 9000)),
        ({"HTTP_HOST": "", "HTTP_PORT": "", "SSE_HOST": "10.0.0.9", "SSE_PORT": "9000"}, ("10.0.0.9", 9000)),
        ({"HTTP_HOST": "", "HTTP_PORT": "", "SSE_HOST": "", "SSE_PORT": ""}, ("0.0.0.0", 80)),
    ],
    ids=["defaults", "http-vars", "http-over-sse", "sse-fallback", "empty-http-falls-back", "all-empty-defaults"],
)
def test_http_bind(_clean_bind_env: None, monkeypatch: pytest.MonkeyPatch,
                   env: dict[str, str], expected: tuple[str, int]) -> None:
    """The bind address follows HTTP_*, then SSE_*, then the defaults, and empty means unset."""
    _set_env(monkeypatch, env)
    assert config.http_bind() == expected


def test_http_bind_rejects_non_numeric_port(_clean_bind_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A port that is not a whole number is an error."""
    monkeypatch.setenv("HTTP_PORT", "eighty")
    with pytest.raises(ValueError):
        config.http_bind()


def test_require_env_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set value comes back as it is."""
    monkeypatch.setenv("X", "http://backend.example.com")
    assert config.require_env("X", "needed by y") == "http://backend.example.com"


@pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_require_env_fails_when_not_set(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                        value: str | None) -> None:
    """An unset, empty or blank value raises and prints one line that names it."""
    if value is None:
        monkeypatch.delenv("X", raising=False)
    else:
        monkeypatch.setenv("X", value)
    with pytest.raises(config.MissingSettingError) as exc_info:
        config.require_env("X", "needed by y")
    assert str(exc_info.value) == "X is not set: needed by y"
    assert capsys.readouterr().err == "[config] X is not set: needed by y\n"


def test_missing_setting_error_is_a_runtime_error() -> None:
    """Code that catches RuntimeError also catches a missing setting."""
    assert issubclass(config.MissingSettingError, RuntimeError)


@pytest.mark.parametrize("url", ["http://h:1", "http://h:1/", "http://h:1/v1", "http://h:1/v1/"])
def test_backend_url_ends_in_one_v1(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """The address gets exactly one /v1, whether or not it was given."""
    monkeypatch.setenv("OCR_BACKEND_URL", url)
    assert config.backend(config.OCR).url == "http://h:1/v1"


def test_backend_carries_model_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set model and key come back with the address."""
    _set_env(monkeypatch, {"OCR_BACKEND_URL": "http://h:1", "OCR_MODEL": "glm-ocr", "OCR_API_KEY": "sk-1"})
    assert config.backend(config.OCR) == config.Backend(url="http://h:1/v1", model="glm-ocr", api_key="sk-1")


@pytest.mark.parametrize(
    "env",
    [{}, {"OCR_MODEL": "", "OCR_API_KEY": ""}],
    ids=["unset", "empty"],
)
def test_backend_model_and_key_are_none_when_not_set(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """An unset or empty model or key comes back as None."""
    monkeypatch.delenv("OCR_MODEL", raising=False)
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    _set_env(monkeypatch, {"OCR_BACKEND_URL": "http://h:1", **env})
    assert config.backend(config.OCR) == config.Backend(url="http://h:1/v1", model=None, api_key=None)


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [("sk-1", {"Authorization": "Bearer sk-1"}), (None, {})],
    ids=["key", "no-key"],
)
def test_backend_auth_headers(api_key: str | None, expected: dict[str, str]) -> None:
    """The key goes in a Bearer header, and no key sends no header."""
    assert config.Backend(url="http://h:1/v1", model=None, api_key=api_key).auth_headers() == expected


def test_backend_first_set_profile_wins(_clean_reason_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With both addresses set, the first profile and its model are used."""
    _set_env(monkeypatch, {
        "REASON_THINKING_BACKEND_URL": "http://think:1", "REASON_THINKING_MODEL": "deep",
        "REASON_BACKEND_URL": "http://plain:1", "REASON_MODEL": "fast",
    })
    result = config.backend(config.REASON_THINKING, config.REASON)
    assert result == config.Backend(url="http://think:1/v1", model="deep", api_key=None)


@pytest.mark.parametrize("first", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_backend_falls_back_to_the_next_profile(_clean_reason_env: None, monkeypatch: pytest.MonkeyPatch,
                                                first: str | None) -> None:
    """When the first address is not set, the next profile is used with its own model and key."""
    if first is not None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", first)
    _set_env(monkeypatch, {
        "REASON_THINKING_MODEL": "deep",
        "REASON_BACKEND_URL": "http://plain:1", "REASON_MODEL": "fast", "REASON_API_KEY": "sk-2",
    })
    result = config.backend(config.REASON_THINKING, config.REASON)
    assert result == config.Backend(url="http://plain:1/v1", model="fast", api_key="sk-2")


def test_backend_fails_when_no_address_is_set(_clean_reason_env: None,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """With no address set, the message names every address tried and the first purpose."""
    message = "REASON_THINKING_BACKEND_URL or REASON_BACKEND_URL is not set: needed by research and browser_use"
    with pytest.raises(config.MissingSettingError) as exc_info:
        config.backend(config.REASON_THINKING, config.REASON)
    assert str(exc_info.value) == message
    assert capsys.readouterr().err == f"[config] {message}\n"


# Each case resolves one research backend, with the three variables it reads.
_RESEARCH_BACKENDS = [
    pytest.param(lambda: config.backend(config.EMBEDDING),
                 ("EMBEDDING_SERVER_URL", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"), id="embedding"),
    pytest.param(lambda: config.backend(config.RERANKER),
                 ("RERANKER_SERVER_URL", "RERANKER_MODEL", "RERANKER_API_KEY"), id="reranker"),
]


@pytest.mark.parametrize(("resolve", "names"), _RESEARCH_BACKENDS)
def test_research_backend_reads_its_own_variables(monkeypatch: pytest.MonkeyPatch,
                                                  resolve: Callable[[], config.Backend],
                                                  names: tuple[str, str, str]) -> None:
    """Each research backend takes its address, model and key from its own variables."""
    url_var, model_var, key_var = names
    _set_env(monkeypatch, {url_var: "http://r:1", model_var: "m-1", key_var: "sk-3"})
    assert resolve() == config.Backend(url="http://r:1/v1", model="m-1", api_key="sk-3")


@pytest.mark.parametrize(("resolve", "names"), _RESEARCH_BACKENDS)
def test_research_backend_fails_without_its_address(monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str],
                                                    resolve: Callable[[], config.Backend],
                                                    names: tuple[str, str, str]) -> None:
    """A research backend with no address names its own variable and research."""
    monkeypatch.delenv(names[0], raising=False)
    message = f"{names[0]} is not set: needed by research"
    with pytest.raises(config.MissingSettingError) as exc_info:
        resolve()
    assert str(exc_info.value) == message
    assert capsys.readouterr().err == f"[config] {message}\n"


def test_tts_backend_takes_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The TTS backend reads its address and model, and never a key."""
    _set_env(monkeypatch, {
        "LOCAL_TTS_URL": "http://tts:1", "LOCAL_TTS_MODEL": "voxcpm2", "LOCAL_TTS_API_KEY": "sk-4",
    })
    assert config.backend(config.TTS) == config.Backend(url="http://tts:1/v1", model="voxcpm2", api_key=None)


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """The module-level names bound to a string literal."""
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants.update({t.id: value.value for t in targets if isinstance(t, ast.Name)})
    return constants


def _read_argument(node: ast.AST) -> ast.expr | None:
    """The name argument of an environment read, or None when node is not one."""
    if isinstance(node, ast.Call) and node.args and ast.unparse(node.func) in _ENV_GETTERS:
        return node.args[0]
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and ast.unparse(node.value) == "os.environ":
        return node.slice
    return None


def _source_reads(root: Path) -> tuple[set[str], list[str]]:
    """Every variable the code under app/ reads, and every read outside config.py whose name is not static."""
    names: set[str] = set()
    dynamic: list[str] = []
    for path in sorted((root / "app").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _string_constants(tree)
        is_config = path.relative_to(root) == _CONFIG_SOURCE
        for node in ast.walk(tree):
            arg = _read_argument(node)
            if arg is None:
                continue
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in constants:
                names.add(constants[arg.id])
            elif not is_config:
                dynamic.append(f"{path.relative_to(root)}:{node.lineno}")
    return names, dynamic


def _profile_vars() -> set[str]:
    """The address, model and key variables of every backend profile."""
    names = set()
    for profile in _PROFILES:
        names.update({profile.url_var, profile.model_var})
        if profile.key_var is not None:
            names.add(profile.key_var)
    return names


def test_env_example_server_part_matches_the_source(env_example: EnvExample,
                                                    request: pytest.FixtureRequest) -> None:
    """Part 1 lists exactly the variables the server code reads."""
    names, dynamic = _source_reads(Path(request.config.rootpath))
    assert dynamic == [], f"environment reads whose name is not a literal or module constant: {dynamic}"
    assert env_example.server == names | _profile_vars()


def test_env_example_lists_each_name_once(env_example: EnvExample) -> None:
    """Each variable has one entry line, and no name is in two parts."""
    names = env_example.names
    assert sorted(n for n in set(names) if names.count(n) > 1) == []
    server, other, host = env_example.server, env_example.other_programs, env_example.host_scripts
    assert server & other == set() and server & host == set() and other & host == set()
    assert other == _OTHER_PROGRAM_VARS
    assert _profile_vars() <= server
