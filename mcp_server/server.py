# mcp_server/server.py
#
# Exposes the agent's codebase tools over MCP, so editors and coding agents (Claude Code, Cursor) can index
# a repository and search and read it through the same hybrid index as the web app and the CLI.

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from api.client import ApiError, RagClient
from mcp_server.backends import LocalBackend
from rag_pipeline.agent import READ_MAX_LINES

INDEX_WAIT_SECONDS = 120.0
SEARCH_K = 6  # the web app's and the CLI's default, so all three show the same sources

INSTRUCTIONS = (
    "Search and read a codebase through a hybrid (semantic + keyword) index. Call index_repo once with a folder "
    "path or a GitHub URL; indexing again later only re-reads changed files. Then use search_code to find code by "
    "meaning, grep for exact names, read_file for exact lines and list_dir for the layout. When more than one "
    "repository is indexed, pass repo (its id, folder path or GitHub URL) to every tool."
)

Backend = RagClient | LocalBackend


def is_github(repo: str) -> bool:
    return repo.startswith(("http://", "https://", "github.com/"))


def describe(repo: dict, job: dict | None = None) -> str:
    line = f"{repo['name']} ({repo['id']}): {repo['status']}, {repo['location']}"
    if repo["stats"]:
        line += f", {repo['stats']['total_files']} files, {repo['stats']['total_chunks']} chunks"
    if job and job["status"] in ("queued", "running"):
        progress = f" {job['done']}/{job['total']}" if job["total"] else ""
        line += f". Job {job['id']} {job['status']}: {job['stage'] or 'waiting'}{progress}"
    elif job and job["status"] == "failed":
        line += f". Last index failed: {job['error']}"
    return line


def create_server(backend: Backend, index_wait: float = INDEX_WAIT_SECONDS) -> MCPServer:
    server = MCPServer("codebase-rag", instructions=INSTRUCTIONS)

    def call(fn, *args, **kwargs):
        """Turns the backend's expected failures into tool errors the model can read and act on."""
        try:
            return fn(*args, **kwargs)
        except (ApiError, ValueError, TimeoutError) as exc:
            raise ToolError(str(exc)) from exc

    def register(repo: str) -> dict:
        return call(backend.add_repo, github_url=repo) if is_github(repo) else call(backend.add_repo, path=repo)

    def find(repo: str | None) -> dict:
        repos = call(backend.repos)
        if repo:
            known = next((r for r in repos if repo in (r["id"], r["location"])), None)
            return known or register(repo)
        if len(repos) == 1:
            return repos[0]
        if not repos:
            raise ToolError("No repository is indexed yet. Call index_repo with a folder path or GitHub URL first.")
        names = ", ".join(f"{r['id']} ({r['location']})" for r in repos)
        raise ToolError(f"Several repositories are indexed; pass repo as one of: {names}")

    def run(repo: str | None, name: str, arguments: dict, k: int | None = None) -> str:
        found = find(repo)
        if found["status"] in ("not_indexed", "failed"):
            raise ToolError(f"{found['name']} is {found['status'].replace('_', ' ')}; call index_repo first.")
        return call(backend.tool, found["id"], name, {key: v for key, v in arguments.items() if v is not None}, k)

    @server.tool()
    def index_repo(repo: str, wait: bool = True) -> str:
        """
        Index a folder (a path on the machine running the index) or a public GitHub repository URL.
        Indexing again re-reads only the files that changed. With wait=false it returns at once; follow the job
        with index_status.
        """
        found = register(repo)
        job = call(backend.index, found["id"])
        if wait:
            try:
                job = backend.wait_for_job(job["id"], timeout=index_wait)
            except TimeoutError:
                job = call(backend.job, job["id"])
        found = call(backend.repo, found["id"])
        if job["status"] == "failed":
            raise ToolError(f"Indexing {found['name']} failed: {job['error']}")
        if job["status"] != "done":
            return f"Indexing {found['name']} started. {describe(found, job)}. Check it with index_status."
        stats = job["stats"]
        changed = f"{stats['files_added']} added, {stats['files_updated']} updated, {stats['files_removed']} removed"
        return (
            f"Indexed {found['name']} ({found['id']}): {stats['total_files']} files, "
            f"{stats['total_chunks']} chunks ({changed})."
        )

    @server.tool()
    def index_status(repo: str | None = None) -> str:
        """Show the indexed repositories, or one repository's status and index job progress."""
        repos = [find(repo)] if repo else call(backend.repos)
        if not repos:
            return "No repositories yet. Call index_repo with a folder path or GitHub URL."
        return "\n".join(describe(r, call(backend.job, r["job_id"]) if r["job_id"] else None) for r in repos)

    @server.tool()
    def search_code(query: str, repo: str | None = None, path: str | None = None, k: int = SEARCH_K) -> str:
        """
        Search the codebase by meaning and keywords. Returns numbered passages with their file and line numbers.
        path limits the search to a folder or file.
        """
        return run(repo, "search_code", {"query": query, "path": path}, max(1, min(k, 20)))

    @server.tool(description=f"Read an indexed file, or a range of its lines (at most {READ_MAX_LINES} per call).")
    def read_file(
        path: str, repo: str | None = None, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        return run(repo, "read_file", {"path": path, "start_line": start_line, "end_line": end_line})

    @server.tool()
    def list_dir(path: str = "", repo: str | None = None) -> str:
        """List the files and folders in a folder of the repository ('' for the root)."""
        return run(repo, "list_dir", {"path": path})

    @server.tool()
    def grep(pattern: str, repo: str | None = None, path: str | None = None) -> str:
        """Find lines matching a regular expression (case-sensitive) in the repository's indexed files."""
        return run(repo, "grep", {"pattern": pattern, "path": path})

    return server
