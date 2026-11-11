"""Settings that more than one part of the server reads, read in one place."""

import os
import sys
from dataclasses import dataclass
from typing import Never

__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_MODEL",
    "EMBEDDING",
    "OCR",
    "REASON",
    "REASON_THINKING",
    "RERANKER",
    "TTS",
    "Backend",
    "MissingSettingError",
    "backend",
    "http_bind",
    "require_env",
]

_DEFAULT_HTTP_HOST = "0.0.0.0"
_DEFAULT_HTTP_PORT = "80"
_API_VERSION_SUFFIX = "/v1"
# What a backend gets when no model or key is set.
DEFAULT_MODEL = "default"
DEFAULT_API_KEY = "not-needed"


@dataclass(frozen=True)
class _Profile:
    """The settings that name one OpenAI-compatible backend.

    Attributes:
        url_var: The variable that holds the server address.
        model_var: The variable that holds the model name.
        key_var: The variable that holds the API key, or None when the
            backend takes no key.
        purpose: What needs the backend, for the message when it is not set.
    """

    url_var: str
    model_var: str
    key_var: str | None
    purpose: str


@dataclass(frozen=True)
class Backend:
    """Where to reach one backend, and how to call it.

    Attributes:
        url: The server address, ending in one /v1.
        model: The model to ask for, or None to let the server choose.
        api_key: The key to send, or None to send none.
    """

    url: str
    model: str | None
    api_key: str | None

    def auth_headers(self) -> dict[str, str]:
        """Build the headers that carry the key.

        Returns:
            An Authorization header when a key is set, or no headers.
        """
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}


REASON = _Profile(
    url_var="REASON_BACKEND_URL",
    model_var="REASON_MODEL",
    key_var="REASON_API_KEY",
    purpose=(
        "needed by google_search, google_flights, web_fetch, read_media_or_doc, "
        "rednote_research, tts and captcha solving"
    ),
)
REASON_THINKING = _Profile(
    url_var="REASON_THINKING_BACKEND_URL",
    model_var="REASON_THINKING_MODEL",
    key_var="REASON_THINKING_API_KEY",
    purpose="needed by research and browser_use",
)
OCR = _Profile(
    url_var="OCR_BACKEND_URL",
    model_var="OCR_MODEL",
    key_var="OCR_API_KEY",
    purpose="needed by OCR of document images",
)
EMBEDDING = _Profile(
    url_var="EMBEDDING_SERVER_URL",
    model_var="EMBEDDING_MODEL",
    key_var="EMBEDDING_API_KEY",
    purpose="needed by research",
)
RERANKER = _Profile(
    url_var="RERANKER_SERVER_URL",
    model_var="RERANKER_MODEL",
    key_var="RERANKER_API_KEY",
    purpose="needed by research",
)
TTS = _Profile(
    url_var="LOCAL_TTS_URL",
    model_var="LOCAL_TTS_MODEL",
    key_var=None,
    purpose="needed by tts and /v1/audio/speech",
)


class MissingSettingError(RuntimeError):
    """A setting that a call needs is not set."""


def _fail(names: tuple[str, ...], purpose: str) -> Never:
    """Print which settings are missing, then raise.

    Args:
        names: The settings that are not set, in the order they were tried.
        purpose: What needs them.

    Raises:
        MissingSettingError: Always.
    """
    message = f"{' or '.join(names)} is not set: {purpose}"
    print(f"[config] {message}", file=sys.stderr, flush=True)
    raise MissingSettingError(message)


def require_env(name: str, purpose: str) -> str:
    """Get a setting that must be set.

    A value that is empty or only spaces counts as not set.

    Args:
        name: The environment variable to read.
        purpose: What needs it, such as "needed by brave_search".

    Returns:
        The value of the setting.

    Raises:
        MissingSettingError: If the setting is not set.
    """
    value = os.getenv(name)
    if not value or not value.strip():
        _fail((name,), purpose)
    return value


def http_bind() -> tuple[str, int]:
    """Get the host and port the server listens on.

    HTTP_HOST and HTTP_PORT come first. SSE_HOST and SSE_PORT are used when
    those are not set. An empty value counts as not set.

    Returns:
        The host and the port.

    Raises:
        ValueError: If the port is not a whole number.
    """
    host = os.getenv("HTTP_HOST") or os.getenv("SSE_HOST") or _DEFAULT_HTTP_HOST
    port = int(os.getenv("HTTP_PORT") or os.getenv("SSE_PORT") or _DEFAULT_HTTP_PORT)
    return host, port


def _optional_env(name: str | None) -> str | None:
    """Get a setting that may be left unset.

    Args:
        name: The environment variable to read, or None for no variable.

    Returns:
        The value, or None when there is no variable or it is unset or empty.
    """
    if name is None:
        return None
    return os.getenv(name) or None


def _with_api_version(url: str) -> str:
    """Make a server address end in exactly one /v1.

    Args:
        url: The address as given, with or without /v1 and a trailing slash.

    Returns:
        The address with trailing slashes removed and one /v1 at the end.
    """
    base = url.strip().rstrip("/")
    if base.endswith(_API_VERSION_SUFFIX):
        base = base[: -len(_API_VERSION_SUFFIX)].rstrip("/")
    return f"{base}{_API_VERSION_SUFFIX}"


def backend(first: _Profile, *fallbacks: _Profile) -> Backend:
    """Get the first backend whose address is set.

    An address that is empty or only spaces counts as not set.

    Args:
        first: The backend to try first.
        *fallbacks: The backends to try next, in order.

    Returns:
        The address, model and key of the first backend with an address.

    Raises:
        MissingSettingError: If none of the backends has an address.
    """
    profiles = (first, *fallbacks)
    for profile in profiles:
        url = os.getenv(profile.url_var)
        if url and url.strip():
            return Backend(
                url=_with_api_version(url),
                model=_optional_env(profile.model_var),
                api_key=_optional_env(profile.key_var),
            )
    _fail(tuple(p.url_var for p in profiles), first.purpose)
