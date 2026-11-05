# mcp_server/__main__.py
#
# Usage (from the project root):
#   python -m mcp_server                               # stdio, for Claude Code and Cursor
#   python -m mcp_server --transport http --port 8001  # streamable HTTP at http://127.0.0.1:8001/mcp
#
# With RAG_API_URL set the tools go through that API; otherwise the index lives in this process.

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load .env before the project is imported: its modules read their defaults (models, paths) at import time.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mcp_server.backends import backend_from_env  # noqa: E402
from mcp_server.server import create_server  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve the codebase tools over MCP.")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio", help="default: stdio")
    parser.add_argument("--host", default="127.0.0.1", help="interface for http (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="port for http (default: 8001)")
    args = parser.parse_args(argv)
    server = create_server(backend_from_env())
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
