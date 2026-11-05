# api/client.py
#
# A small client for the API, so the CLI (--api) and the Streamlit app (RAG_API_URL) can index and ask
# through one shared service instead of loading the models themselves.

import json
import time
from collections.abc import Callable, Iterator
from itertools import chain

import httpx

from rag_pipeline.pipeline import Answer, Sources, Step, Token
from vectorstore.vector_store import StoredChunk

DEFAULT_API_URL = "http://127.0.0.1:8000"
CHAT_OPTIONS = ("provider", "model", "k", "rerank", "rewrite_followups", "mode")  # see api.schemas.ChatIn


class ApiError(RuntimeError):
    """The API refused a request, or failed while streaming an answer."""


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:
    """Yields (event, data) for each server-sent event; data lines are JSON."""
    event, data = "message", []
    for line in chain(lines, [""]):
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        # anything else (": ping" comments, id:, retry:) carries nothing we use


def source_chunk(source: dict) -> StoredChunk:
    """Turns a source sent by the API back into the chunk the in-process pipeline would return."""
    meta = {key: source[key] for key in ("source", "start_line", "end_line", "symbol", "language")}
    meta["header_lines"] = 0  # the API sends the code without the File/Symbol header
    return StoredChunk(f"{meta['source']}:{meta['start_line']}-{meta['end_line']}", source["code"], meta)


class RagClient:
    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout: float = 30.0,
        poll_seconds: float = 0.5,
        http: httpx.Client | None = None,
    ):
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.timeout = timeout
        self.poll_seconds = poll_seconds
        self.last_done: dict | None = None  # the done event of the last answer: provider, model, full text

    def _request(self, method: str, url: str, **kwargs):
        response = self.http.request(method, url, **kwargs)
        if response.is_error:
            raise ApiError(error_message(response))
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def add_repo(self, path: str | None = None, github_url: str | None = None) -> dict:
        """Registers a folder on the API's machine or a public GitHub repo; registering twice is harmless."""
        return self._request(
            "POST", "/repos", json={"path": path} if github_url is None else {"github_url": github_url}
        )

    def repos(self) -> list[dict]:
        return self._request("GET", "/repos")

    def repo(self, repo_id: str) -> dict:
        return self._request("GET", f"/repos/{repo_id}")

    def index(self, repo_id: str) -> dict:
        """Starts an index job (or returns the one already running)."""
        return self._request("POST", f"/repos/{repo_id}/index")

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def wait_for_job(
        self, job_id: str, on_progress: Callable[[dict], None] | None = None, timeout: float | None = None
    ) -> dict:
        """Polls a job until it is done or failed, passing each record to on_progress."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if on_progress:
                on_progress(job)
            if job["status"] not in ("queued", "running"):
                return job
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Index job {job_id} is still {job['status']}")
            time.sleep(self.poll_seconds)

    def search(self, repo_id: str, query: str, k: int = 6) -> list[StoredChunk]:
        sources = self._request("POST", f"/repos/{repo_id}/search", json={"query": query, "k": k})
        return [source_chunk(source) for source in sources]

    def tool(self, repo_id: str, name: str, arguments: dict, k: int | None = None) -> str:
        """Runs one of the agent's tools on the API and returns its text result."""
        params = {} if k is None else {"k": k}
        return self._request("POST", f"/repos/{repo_id}/tools/{name}", json=arguments, params=params)["result"]

    def chat_events(
        self, repo_id: str, question: str, history: list[dict] | None = None, **options
    ) -> Iterator[Step | Sources | Token]:
        """
        The same events RAGPipeline.answer_events yields, read from the API's stream.

        options are the API's chat settings (CHAT_OPTIONS); any left out take the server's defaults.
        """
        unknown = set(options) - set(CHAT_OPTIONS)
        if unknown:
            raise TypeError(f"Unknown chat options: {', '.join(sorted(unknown))}")
        body = {"question": question, "history": history or [], **options}
        self.last_done = None
        # A local model can think for minutes between events, so only the connection has a time limit.
        timeout = httpx.Timeout(self.timeout, read=None)
        with self.http.stream("POST", f"/repos/{repo_id}/chat", json=body, timeout=timeout) as response:
            if response.is_error:
                response.read()
                raise ApiError(error_message(response))
            for event, data in parse_sse(response.iter_lines()):
                if event == "step":
                    yield Step(data["label"])
                elif event == "sources":
                    yield Sources(data["search_query"], [source_chunk(source) for source in data["sources"]])
                elif event == "token":
                    yield Token(data["text"])
                elif event == "error":
                    raise ApiError(data["message"])
                elif event == "done":
                    self.last_done = data
                    return
        raise ApiError("The answer stream ended before it was done")

    def ask(self, repo_id: str, question: str, *args, on_step: Callable[[str], None] | None = None, **kwargs) -> Answer:
        """Like RAGPipeline.stream_answer: waits for the sources and returns them with the reply still to stream."""
        step = on_step or (lambda label: None)
        events = self.chat_events(repo_id, question, *args, **kwargs)
        for event in events:
            if isinstance(event, Sources):
                break
            step(event.label)
        else:
            raise ApiError("The answer stream ended before its sources")
        answer = Answer(question, event.search_query, event.chunks)
        first = next(events, None)  # nothing after the sources means retrieval-only mode
        if first is not None:
            answer.stream = replies(chain([first], events), step)
        return answer


def replies(events: Iterator[Step | Token], step: Callable[[str], None]) -> Iterator[str]:
    for event in events:
        if isinstance(event, Step):
            step(event.label)
        else:
            yield event.text


def error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return f"{detail if isinstance(detail, str) else response.text} (HTTP {response.status_code})"


def connect(base_url: str = DEFAULT_API_URL) -> RagClient:
    return RagClient(base_url)
