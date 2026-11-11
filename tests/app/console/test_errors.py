"""Tests for the console error responses."""
from __future__ import annotations

import json

from starlette.responses import JSONResponse

from app.console.errors import error_response


def test_error_response_carries_status_code_and_envelope() -> None:
    r = error_response(404, "not_found", "Unknown tool 'x'.")
    assert isinstance(r, JSONResponse)
    assert r.status_code == 404
    assert json.loads(r.body) == {"error": "Unknown tool 'x'.", "code": "not_found"}


def test_error_response_applies_headers_only_when_given() -> None:
    r = error_response(405, "method_not_allowed", "The method is not allowed.", {"Allow": "GET, HEAD"})
    assert r.headers["allow"] == "GET, HEAD"
    assert "allow" not in error_response(500, "internal", "The tool call failed.").headers
