"""Starts the server.

HTTP_HOST and HTTP_PORT say where it listens. SSE_HOST and SSE_PORT are used
when those are not set. Without any of them it listens on 0.0.0.0 port 80.
"""

import uvicorn

from app.lib.shared import config
from app.server import build_app

if __name__ == "__main__":
    _host, _port = config.http_bind()
    uvicorn.run(build_app(), host=_host, port=_port)
