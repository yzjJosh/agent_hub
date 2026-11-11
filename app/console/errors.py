"""The error reply the console API sends."""
from __future__ import annotations

from starlette.responses import JSONResponse

__all__ = ["error_response"]


def error_response(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    """Build an error reply for the console API.

    Args:
        status: The HTTP status code to send.
        code: A short name for the error.
        message: What went wrong, in words.
        headers: Extra headers to send, or None.

    Returns:
        The error reply.
    """
    return JSONResponse({"error": message, "code": code}, status_code=status, headers=headers)
