from __future__ import annotations

import argparse
import hmac
import os

from mcp.server.mcpserver import MCPServer

from llmgw import load_env

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class BearerAuth:
    """ASGI middleware: every HTTP request must carry ``Authorization: Bearer <token>``."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"missing or wrong bearer token"})
                return
        await self.app(scope, receive, send)


def http_app(server: MCPServer, host: str, token: str | None):
    """The streamable-HTTP app at /mcp, behind bearer auth when ``token`` is set. Refuses to
    serve beyond localhost without a token: the tools expose your documents and eval data."""
    if not token and host not in _LOOPBACK:
        raise SystemExit(f"Refusing to serve on {host} without MCP_AUTH_TOKEN: anyone on the network could "
                         "read your documents. Set MCP_AUTH_TOKEN, or keep --host 127.0.0.1.")
    app = server.streamable_http_app(host=host)
    return BearerAuth(app, token) if token else app


def run(server: MCPServer, argv: list[str] | None = None) -> None:
    """stdio by default (what Claude Code / Desktop launch); ``--http`` serves streamable HTTP
    at /mcp, requiring ``Authorization: Bearer $MCP_AUTH_TOKEN`` when that variable is set."""
    p = argparse.ArgumentParser(prog=server.name)
    p.add_argument("--http", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args, _ = p.parse_known_args(argv)
    load_env()  # RETRIEVAL_INDEX_DIR, EVALH_DB, PROMPTS_DIR, MCP_AUTH_TOKEN and keys can live in .env
    if args.http:
        import uvicorn

        uvicorn.run(http_app(server, args.host, os.getenv("MCP_AUTH_TOKEN") or None), host=args.host, port=args.port)
    else:
        server.run()
