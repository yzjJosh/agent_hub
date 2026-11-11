"""Endpoint that renders one message's bytes through the generic pipeline."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.console.errors import error_response
from app.lib.render.classify import kind_label
from app.lib.render.detect import TEXT_FAMILY_KINDS, detect
from app.lib.render.fragments import build_fragment_bytes

__all__ = ["render_message_impl"]

_ABSOLUTE_CAP = 4 * 1024 * 1024
_NO_PREVIEW_NOTE = "No preview available for this content."


async def render_message_impl(request: Request) -> Response:
    """Render one assistant message's bytes through the generic pipeline.

    Args:
        request: The HTTP request, whose body is the message bytes.

    Returns:
        The descriptor-shaped payload, or an error reply.
    """
    body = await request.body()
    if len(body) > _ABSOLUTE_CAP:
        return error_response(400, "bad_param", "Message body exceeds the 4 MiB request cap.")
    detection = await detect(body)
    kind = detection.kind
    if kind not in TEXT_FAMILY_KINDS:
        fragment, meta = None, {"mode": "raw", "note": _NO_PREVIEW_NOTE}
    else:
        fragment, meta = await build_fragment_bytes(kind, body, detection.language)
    return JSONResponse({
        "kind": kind.value,
        "kind_label": kind_label(kind, "", detection.language),
        "mode": meta.get("mode"),
        "fragment": fragment,
        "note": meta.get("note"),
        "size": len(body),
    })
