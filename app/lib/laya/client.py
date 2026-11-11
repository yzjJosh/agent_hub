"""Async HTTP client for the Laya page classifier.

The base address comes from the LAYA_SERVER_URL environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.lib.shared import config

__all__ = ["LayaError", "Verdict", "classify"]

_SERVER_URL_ENV = "LAYA_SERVER_URL"
_SERVER_PURPOSE = "needed by web_fetch and read_media_or_doc to check web pages"
_REQUEST_TIMEOUT_S = 60
_CONNECT_TIMEOUT_S = 10
_REQUEST_TIMEOUT = httpx.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
_HTTP_OK = 200
_BODY_SNIPPET_LEN = 200


class LayaError(Exception):
    """Laya gave no usable answer."""


@dataclass(frozen=True)
class Verdict:
    """What Laya says about one page.

    Attributes:
        blocked: True when the page is a block page and not real content.
        blocked_probability: How sure Laya is that the page is blocked, from 0 to 1.
    """

    blocked: bool
    blocked_probability: float


def _server_url() -> str:
    """The Laya base address to talk to."""
    return config.require_env(_SERVER_URL_ENV, _SERVER_PURPOSE).rstrip("/")


async def _post_html(url: str, html: str) -> httpx.Response:
    """Send the page to Laya."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_REQUEST_TIMEOUT) as client:
            return await client.post(url, json={"html": html})
    except httpx.HTTPError as exc:
        raise LayaError(f"could not reach Laya: {exc}") from exc


def _parse_verdict(response: httpx.Response) -> Verdict:
    """The verdict in a 200 answer."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise LayaError(f"Laya sent a body that is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LayaError("Laya sent a body that is not a JSON object")
    blocked = payload.get("blocked")
    probability = payload.get("blocked_probability")
    if not isinstance(blocked, bool):
        raise LayaError(f"Laya sent no true or false blocked field: {blocked!r}")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise LayaError(f"Laya sent no number blocked_probability field: {probability!r}")
    return Verdict(blocked=blocked, blocked_probability=float(probability))


async def classify(html: str) -> Verdict:
    """Ask Laya whether a page is a block page.

    Args:
        html: The page HTML.

    Returns:
        Whether the page is blocked, and how sure Laya is.

    Raises:
        LayaError: If Laya cannot be reached, turns the page down, or sends
            a bad answer.
        config.MissingSettingError: If LAYA_SERVER_URL is not set.
    """
    response = await _post_html(f"{_server_url()}/v1/classify", html)
    if response.status_code != _HTTP_OK:
        body = response.text[:_BODY_SNIPPET_LEN]
        raise LayaError(f"Laya rejected the page: HTTP {response.status_code} {body}")
    return _parse_verdict(response)
