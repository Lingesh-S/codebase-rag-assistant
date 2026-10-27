from langchain_core.documents import Document

from utils.chunker import CodeChunker, chunk_body, chunk_documents


def make_doc(text: str, language: str = "python", source: str = "pkg/module.py") -> Document:
    return Document(page_content=text, metadata={"source": source, "language": language, "file_hash": "abc"})


def function_source(name: str, body_lines: int = 12) -> str:
    body = "\n".join(f"    value_{i} = '{name}' * {i}" for i in range(body_lines))
    return f"def {name}(arg):\n    \"\"\"Docstring for {name}.\"\"\"\n{body}\n    return arg\n"


def assert_body_matches_lines(chunk: Document, source_text: str) -> None:
    lines = source_text.splitlines()
    meta = chunk.metadata
    expected = "\n".join(lines[meta["start_line"] - 1:meta["end_line"]])
    assert chunk_body(chunk.page_content, meta["header_lines"]) == expected


def test_small_file_is_one_chunk_with_header_and_symbols():
    text = "import os\n\n\ndef greet(name):\n    return f'hi {name}'\n\n\nclass Greeter:\n    pass\n"
    [chunk] = chunk_documents([make_doc(text)])

    assert chunk.metadata["start_line"] == 1
    assert chunk.metadata["end_line"] == 9
    assert chunk.metadata["symbol"] == "greet, Greeter"
    assert chunk.page_content.startswith("File: pkg/module.py (lines 1-9)\nSymbol: greet, Greeter\n\n")
    assert_body_matches_lines(chunk, text)


def test_large_python_file_splits_on_function_boundaries():
    names = ["load_config", "parse_args", "run_server", "shutdown"]
    text = "import os\nimport sys\n\n\n" + "\n\n".join(function_source(n) for n in names)
    chunks = CodeChunker(chunk_size=600, chunk_overlap=0).split(make_doc(text))

    symbols = [c.metadata["symbol"] for c in chunks]
    for name in names:
        assert any(name in s for s in symbols)
        # each function lives entirely inside one chunk
        assert sum(f"def {name}(" in c.page_content for c in chunks) == 1
    for chunk in chunks:
        assert_body_matches_lines(chunk, text)


def test_large_class_is_split_into_methods_with_qualified_symbols():
    methods = "\n".join(
        "    " + line if line else line
        for name in ["save", "load", "delete"]
        for line in function_source(name).replace("(arg)", "(self, arg)").splitlines() + [""]
    )
    text = 'class Repository:\n    """Stores records."""\n\n    table = "records"\n\n' + methods
    chunks = CodeChunker(chunk_size=600, chunk_overlap=0).split(make_doc(text))

    symbols = [c.metadata["symbol"] for c in chunks]
    # the short class header is merged into the first method's chunk
    assert symbols[0].split(", ")[0] == "Repository"
    assert 'table = "records"' in chunks[0].page_content
    for method in ["save", "load", "delete"]:
        assert any(f"Repository.{method}" in s for s in symbols)
    for chunk in chunks:
        assert_body_matches_lines(chunk, text)


def test_every_non_blank_python_line_lands_in_a_chunk():
    methods = "\n".join("    " + line if line else line for line in function_source("process").replace("(arg)", "(self, arg)").splitlines())
    text = (
        '"""Module docstring."""\nimport os\n\n# --- helpers ---\n\n'
        + "\n\n".join(function_source(f"fn_{i}") for i in range(3))
        + '\n\n\nclass Big:\n    """Class docstring."""\n\n    LIMIT = 3\n\n'
        + methods
        + "\n\n# end of module\n"
    )
    chunks = CodeChunker(chunk_size=600, chunk_overlap=0).split(make_doc(text))

    covered = {n for c in chunks for n in range(c.metadata["start_line"], c.metadata["end_line"] + 1)}
    non_blank = {i for i, line in enumerate(text.splitlines(), 1) if line.strip()}
    assert non_blank <= covered
    [header] = [c for c in chunks if "Class docstring." in c.page_content]
    assert "Big" in header.metadata["symbol"].split(", ")


def test_decorators_and_leading_comments_stay_with_their_function():
    filler = "\n\n".join(function_source(f"helper_{i}") for i in range(3))
    text = filler + "\n\n# Cache results because parsing is slow.\n@functools.cache\ndef parse(path):\n    return open(path).read()\n"
    chunks = CodeChunker(chunk_size=600, chunk_overlap=0).split(make_doc(text))

    [parse_chunk] = [c for c in chunks if "def parse(" in c.page_content]
    body = chunk_body(parse_chunk.page_content, parse_chunk.metadata["header_lines"])
    assert "# Cache results because parsing is slow.\n@functools.cache\ndef parse(path):" in body


def test_invalid_python_falls_back_to_text_splitting():
    text = "def broken(:\n" + "\n".join(f"    x_{i} = {i}" for i in range(200))
    chunks = CodeChunker(chunk_size=500, chunk_overlap=50).split(make_doc(text))

    assert len(chunks) > 1
    for chunk in chunks:
        assert 1 <= chunk.metadata["start_line"] <= chunk.metadata["end_line"] <= len(text.splitlines())
        assert_body_matches_lines(chunk, text)


def test_other_languages_get_line_ranges():
    text = "\n\n".join(f"function handler{i}(req, res) {{\n  const value = compute({i});\n  res.send(value);\n}}" for i in range(40))
    chunks = CodeChunker(chunk_size=400, chunk_overlap=0).split(make_doc(text, "js", "web/handlers.js"))

    assert len(chunks) > 1
    assert chunks[0].metadata["start_line"] == 1
    assert chunks[-1].metadata["end_line"] == len(text.splitlines())
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.metadata["start_line"], chunk.metadata["end_line"] + 1))
        assert_body_matches_lines(chunk, text)
    non_blank = {i for i, line in enumerate(text.splitlines(), 1) if line.strip()}
    assert non_blank <= covered


def test_chunk_ids_are_deterministic_and_unique():
    text = "\n\n".join(function_source(f"fn_{i}") for i in range(6))
    first = CodeChunker(chunk_size=600).split(make_doc(text))
    second = CodeChunker(chunk_size=600).split(make_doc(text))

    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
