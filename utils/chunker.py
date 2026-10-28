# utils/chunker.py

import ast
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 300

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

# A span is (start_line, end_line, symbol); lines are 1-based and inclusive.
Span = tuple[int, int, str]


class CodeChunker:
    """
    Splits source files into retrieval-sized chunks that follow code structure.

    Python is split along the AST so functions, classes and methods stay intact.
    Other languages use LangChain's language-aware separators. Every chunk keeps
    its line range and symbol name, and gets a short "File / Symbol" header so the
    embedding and the LLM both see where the code lives.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self._splitters: dict[str, RecursiveCharacterTextSplitter] = {}

    def split(self, doc: Document) -> list[Document]:
        text = doc.page_content
        lines = text.splitlines()
        language = doc.metadata.get("language", "text")

        spans = None
        if len(text) <= self.chunk_size:
            spans = self._fit(lines, 1, len(lines), self._top_level_symbols(text, language))
        elif language == "python":
            spans = self._python_spans(text, lines)
        if spans is None:
            spans = self._splitter_spans(text, lines, language, symbol="")

        return [self._make_chunk(doc, lines, span, i) for i, span in enumerate(spans)]

    # ---- Python -------------------------------------------------------------

    def _python_spans(self, text: str, lines: list[str]) -> list[Span] | None:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return None
        spans = self._body_spans(lines, tree.body, parent="", floor=1)
        # Keep trailing comments at the end of the module with the last chunk.
        last_line = max((i for i, line in enumerate(lines, 1) if line.strip()), default=0)
        if spans and spans[-1][1] < last_line:
            spans[-1] = (spans[-1][0], last_line, spans[-1][2])
        return self._merge_small(lines, spans)

    def _body_spans(self, lines: list[str], nodes: list[ast.stmt], parent: str, floor: int) -> list[Span]:
        spans: list[Span] = []
        block_start = block_end = None
        for node in nodes:
            start = self._node_start(lines, node, floor)
            if isinstance(node, DEFINITIONS):
                if block_start is not None:
                    spans += self._fit(lines, block_start, block_end, parent)
                    block_start = None
                spans += self._definition_spans(lines, node, parent, start)
            else:
                if block_start is None:
                    block_start = start
                block_end = node.end_lineno
            floor = node.end_lineno + 1
        if block_start is not None:
            spans += self._fit(lines, block_start, block_end, parent)
        return spans

    def _definition_spans(self, lines: list[str], node, parent: str, start: int) -> list[Span]:
        name = f"{parent}.{node.name}" if parent else node.name
        end = node.end_lineno
        if self._size(lines, start, end) <= self.chunk_size or not isinstance(node, ast.ClassDef):
            return self._fit(lines, start, end, name)

        # Large class: signature, docstring and attributes form one chunk, then
        # each method is chunked on its own under "Class.method".
        first = next((i for i, n in enumerate(node.body) if isinstance(n, DEFINITIONS)), None)
        if first is None:
            return self._fit(lines, start, end, name)
        floor = node.body[first - 1].end_lineno + 1 if first else start + 1
        first_child_start = self._node_start(lines, node.body[first], floor)
        spans = self._fit(lines, start, first_child_start - 1, name)
        return spans + self._body_spans(lines, node.body[first:], name, first_child_start)

    @staticmethod
    def _node_start(lines: list[str], node, floor: int) -> int:
        """
        First line of a node: its decorators, plus any comments between it and the
        previous statement (floor is the line right after that statement).
        """
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        if any(line.strip() for line in lines[floor - 1 : start - 1]):
            start = floor
        return start

    def _merge_small(self, lines: list[str], spans: list[Span]) -> list[Span]:
        """Joins neighbouring tiny spans (one-line helpers, import blocks) so chunks keep enough context."""
        merged: list[Span] = []
        for start, end, symbol in spans:
            if merged:
                p_start, p_end, p_symbol = merged[-1]
                small = min(self._size(lines, p_start, p_end), self._size(lines, start, end)) < MIN_CHUNK_CHARS
                if small and self._size(lines, p_start, end) <= self.chunk_size:
                    symbols = [s for s in dict.fromkeys(p_symbol.split(", ") + symbol.split(", ")) if s]
                    merged[-1] = (p_start, end, ", ".join(symbols))
                    continue
            merged.append((start, end, symbol))
        return merged

    @staticmethod
    def _top_level_symbols(text: str, language: str) -> str:
        if language != "python":
            return ""
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return ""
        return ", ".join(n.name for n in tree.body if isinstance(n, DEFINITIONS))

    # ---- Generic splitting ----------------------------------------------------

    def _fit(self, lines: list[str], start: int, end: int, symbol: str) -> list[Span]:
        """Trims blank edge lines and splits the range further if it is still too large."""
        while start <= end and not lines[start - 1].strip():
            start += 1
        while end >= start and not lines[end - 1].strip():
            end -= 1
        if start > end:
            return []
        if self._size(lines, start, end) <= self.chunk_size:
            return [(start, end, symbol)]
        text = "\n".join(lines[start - 1 : end])
        return [
            (start + s - 1, start + e - 1, symbol)
            for s, e, _ in self._splitter_spans(text, text.splitlines(), "python", symbol)
        ]

    def _splitter_spans(self, text: str, lines: list[str], language: str, symbol: str) -> list[Span]:
        splitter = self._splitter(language)
        spans = []
        for part in splitter.create_documents([text]):
            offset = part.metadata.get("start_index", -1)
            if offset < 0:
                continue
            start = text.count("\n", 0, offset) + 1
            end = start + part.page_content.count("\n")
            spans.append((start, min(end, len(lines)), symbol))
        return spans

    def _splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        if language not in self._splitters:
            kwargs = {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap, "add_start_index": True}
            try:
                splitter = RecursiveCharacterTextSplitter.from_language(Language(language), **kwargs)
            except ValueError:
                splitter = RecursiveCharacterTextSplitter(**kwargs)
            self._splitters[language] = splitter
        return self._splitters[language]

    @staticmethod
    def _size(lines: list[str], start: int, end: int) -> int:
        return sum(len(line) + 1 for line in lines[start - 1 : end])

    # ---- Output -------------------------------------------------------------------

    @staticmethod
    def _make_chunk(doc: Document, lines: list[str], span: Span, index: int) -> Document:
        start, end, symbol = span
        source = doc.metadata["source"]
        header = [f"File: {source} (lines {start}-{end})"]
        if symbol:
            header.append(f"Symbol: {symbol}")
        body = "\n".join(lines[start - 1 : end])
        chunk_id = hashlib.sha1(
            f"{source}:{doc.metadata.get('file_hash', '')}:{start}:{end}:{index}".encode(),
            usedforsecurity=False,  # a stable ID, not a security boundary
        ).hexdigest()
        return Document(
            id=chunk_id,
            page_content="\n".join(header) + "\n\n" + body,
            metadata={
                **doc.metadata,
                "start_line": start,
                "end_line": end,
                "symbol": symbol,
                "chunk_index": index,
                "header_lines": len(header) + 1,
            },
        )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    chunker = CodeChunker(chunk_size, chunk_overlap)
    return [chunk for doc in documents for chunk in chunker.split(doc)]


def chunk_body(chunk_text: str, header_lines: int) -> str:
    """Strips the File/Symbol header added by CodeChunker, leaving only the source code."""
    return "\n".join(chunk_text.split("\n")[header_lines:])
