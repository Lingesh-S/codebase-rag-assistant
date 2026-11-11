# api/__main__.py
#
# Usage (from the project root):
#   python -m api                      # http://127.0.0.1:8000, interactive docs at /docs
#   python -m api --host 0.0.0.0 --port 9000

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load .env before the app is imported: project modules read their defaults (models, paths) at import time.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import uvicorn  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve the codebase RAG API.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    args = parser.parse_args(argv)
    uvicorn.run("api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
