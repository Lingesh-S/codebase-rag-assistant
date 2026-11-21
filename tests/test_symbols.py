import pytest
from langchain_core.documents import Document

from utils.chunker import CodeChunker, chunk_body
from utils.symbols import Symbol, extract_symbols

# language -> (file suffix, source, expected dotted symbol names in order)
FIXTURES = {
    "js": (
        ".js",
        "import x from 'y';\n\n/** Adds one. */\nconst inc = (a) => a + 1;\n\nfunction plain() {\n  return 1;\n}\n\n"
        "export class Box extends Base {\n  open() {}\n  static make() {}\n}\n\nexport default function main() {}\n",
        ["inc", "plain", "Box", "Box.open", "Box.make", "main"],
    ),
    "ts": (
        ".ts",
        "interface Shape { area: number }\ntype Id = string;\nenum Color { Red }\n"
        "abstract class Base { abstract run(): void; stop(): void {} }\nexport const load = async () => 1;\n",
        ["Shape", "Id", "Color", "Base", "Base.stop", "load"],
    ),
    "tsx": (
        ".tsx",
        "const App = () => <div />;\nfunction Page() {\n  return <p>hi</p>;\n}\n",
        ["App", "Page"],
    ),
    "java": (
        ".java",
        "package a;\nimport b.C;\n\npublic class Account {\n  int total;\n  Account() {}\n"
        "  public void deposit() {}\n}\n"
        "interface Store { void put(); }\nenum Kind { A }\nrecord Pair(int a, int b) {}\n",
        ["Account", "Account.Account", "Account.deposit", "Store", "Store.put", "Kind", "Pair"],
    ),
    "go": (
        ".go",
        'package a\n\nimport "fmt"\n\ntype Server struct{ Port int }\n\nfunc New() *Server { return nil }\n\n'
        "func (s *Server) Start() {}\n",
        ["Server", "New", "Server.Start"],
    ),
    "rust": (
        ".rs",
        "use std::fmt;\n\nstruct Point { x: i32 }\n\ntrait Shape { fn area(&self) -> i32; }\n\n"
        "impl Point {\n    fn new() -> Point { Point { x: 0 } }\n}\n\n"
        "mod util {\n    pub fn help() {}\n}\n\nfn main() {}\n",
        ["Point", "Shape", "Shape.area", "Point", "Point.new", "util", "util.help", "main"],
    ),
    "c": (
        ".c",
        "#include <stdio.h>\n\nstruct node { int v; };\n\nstatic int *make(int a) { return 0; }\n\n"
        "int main(void) { return 0; }\n",
        ["node", "make", "main"],
    ),
    "cpp": (
        ".cpp",
        "#include <a>\n\nnamespace shop {\nclass Cart {\npublic:\n  int count() { return 1; }\n};\n"
        "template <typename T> T id(T x) { return x; }\n}\n\nint Cart::total() { return 2; }\n",
        ["shop", "shop.Cart", "shop.Cart.count", "shop.id", "Cart::total"],
    ),
    "csharp": (
        ".cs",
        "using System;\n\nnamespace Shop {\n  public class Cart {\n    public int Count { get; set; }\n"
        "    public Cart() {}\n    public void Add() {}\n  }\n  interface IStore {}\n}\n",
        ["Shop", "Shop.Cart", "Shop.Cart.Count", "Shop.Cart.Cart", "Shop.Cart.Add", "Shop.IStore"],
    ),
    "ruby": (
        ".rb",
        "require 'a'\n\nmodule Shop\n  class Cart\n    def add; end\n    def self.build; end\n  end\nend\n\n"
        "def helper; end\n",
        ["Shop", "Shop.Cart", "Shop.Cart.add", "Shop.Cart.build", "helper"],
    ),
    "php": (
        ".php",
        "<?php\nnamespace App;\n\nfunction helper() {}\n\nclass Cart {\n  public function add() {}\n}\n\n"
        "interface Store {}\ntrait Loggable {}\n",
        ["helper", "Cart", "Cart.add", "Store", "Loggable"],
    ),
    "kotlin": (
        ".kt",
        "package a\n\nclass Cart {\n  fun add() {}\n}\n\nobject Registry { fun get() {} }\n\n"
        "interface Store { fun put() }\n\nfun top() {}\n",
        ["Cart", "Cart.add", "Registry", "Registry.get", "Store", "Store.put", "top"],
    ),
}

# one definition per language, used to build files big enough to be split
PADDED = {
    "js": ("function {name}() {{\n{body}\n}}\n", "  const value_{i} = {i};"),
    "ts": ("function {name}(): void {{\n{body}\n}}\n", "  const value_{i}: number = {i};"),
    "tsx": ("function {name}() {{\n{body}\n}}\n", "  const value_{i} = {i};"),
    "java": ("class {name} {{\n  void run() {{\n{body}\n  }}\n}}\n", "    int value_{i} = {i};"),
    "go": ("func {name}() {{\n{body}\n}}\n", "\tvalue_{i} := {i}"),
    "rust": ("fn {name}() {{\n{body}\n}}\n", "    let value_{i} = {i};"),
    "c": ("int {name}(void) {{\n{body}\n    return 0;\n}}\n", "    int value_{i} = {i};"),
    "cpp": ("int {name}() {{\n{body}\n    return 0;\n}}\n", "    int value_{i} = {i};"),
    "csharp": ("class {name} {{\n  void Run() {{\n{body}\n  }}\n}}\n", "    int value_{i} = {i};"),
    "ruby": ("def {name}\n{body}\nend\n", "  value_{i} = {i}"),
    "php": ("function {name}() {{\n{body}\n}}\n", "  $value_{i} = {i};"),
    "kotlin": ("fun {name}() {{\n{body}\n}}\n", "  val value_{i} = {i}"),
}


def dotted(symbols: list[Symbol], parent: str = "") -> list[str]:
    names = []
    for s in symbols:
        name = f"{parent}.{s.name}" if parent else s.name
        names += [name, *dotted(s.children, name)]
    return names


@pytest.mark.parametrize("language", FIXTURES)
def test_extract_symbols_names_nesting_and_order(language):
    suffix, source, expected = FIXTURES[language]
    symbols = extract_symbols(source, language, suffix)
    assert symbols is not None
    assert dotted(symbols) == expected


def test_symbol_lines_are_one_based_and_include_leading_comment():
    suffix, source, _ = FIXTURES["js"]
    symbols = extract_symbols(source, "js", suffix)
    inc = symbols[0]
    # the doc comment on line 3 belongs to the definition on line 4
    assert (inc.start, inc.end) == (3, 4)
    plain = symbols[1]
    assert (plain.start, plain.end) == (6, 8)
    box = symbols[2]
    assert (box.start, box.end) == (10, 13)
    assert [(c.start, c.end) for c in box.children] == [(11, 11), (12, 12)]


def test_go_method_is_named_after_its_receiver():
    suffix, source, _ = FIXTURES["go"]
    names = [s.name for s in extract_symbols(source, "go", suffix)]
    assert names[-1] == "Server.Start"


def test_unsupported_language_and_syntax_errors_return_none():
    assert extract_symbols("a = 1", "markdown", ".md") is None
    assert extract_symbols("function ( {{{ ;;", "js", ".js") is None


def test_text_with_exotic_line_breaks_returns_none():
    # tree-sitter counts rows by \n only, so line numbers could drift
    assert extract_symbols("function a() {}\x0c\nfunction b() {}\n", "js", ".js") is None


def make_doc(text: str, language: str, suffix: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": f"src/file{suffix}", "language": language, "file_hash": "h"},
    )


@pytest.mark.parametrize("language", PADDED)
def test_chunks_follow_symbol_boundaries(language):
    template, line = PADDED[language]
    suffix = FIXTURES[language][0]
    names = ["alpha_task", "beta_task", "gamma_task"]
    body = "\n".join(line.format(i=i) for i in range(22))
    text = "\n\n".join(template.format(name=n, body=body) for n in names)
    if language == "php":
        text = "<?php\n" + text
    chunks = CodeChunker(chunk_size=900, chunk_overlap=0).split(make_doc(text, language, suffix))

    assert len(chunks) >= 3
    lines = text.splitlines()
    for chunk in chunks:
        meta = chunk.metadata
        assert chunk_body(chunk.page_content, meta["header_lines"]) == "\n".join(
            lines[meta["start_line"] - 1 : meta["end_line"]]
        )
    for name in names:
        owners = [c for c in chunks if name in c.metadata["symbol"]]
        assert len(owners) == 1, f"{name} should sit in exactly one chunk"
        # the whole definition (first to last line) is inside that chunk
        first = next(i for i, ln in enumerate(lines, 1) if name in ln)
        assert owners[0].metadata["start_line"] <= first
        assert owners[0].metadata["end_line"] >= first + 20


def test_large_java_class_splits_into_qualified_methods():
    body = "\n".join(f"    int v{i} = {i};" for i in range(12))
    methods = "\n".join(f"  void {n}() {{\n{body}\n  }}\n" for n in ["save", "load", "delete"])
    text = f"public class Repo {{\n  int count;\n\n{methods}}}\n"
    chunks = CodeChunker(chunk_size=600, chunk_overlap=0).split(make_doc(text, "java", ".java"))

    symbols = [c.metadata["symbol"] for c in chunks]
    for method in ["save", "load", "delete"]:
        assert any(f"Repo.{method}" in s for s in symbols)
    covered = {n for c in chunks for n in range(c.metadata["start_line"], c.metadata["end_line"] + 1)}
    assert all(n in covered for n, line in enumerate(text.splitlines(), 1) if line.strip())


def test_small_non_python_file_lists_its_top_level_symbols():
    suffix, source, _ = FIXTURES["go"]
    [chunk] = CodeChunker().split(make_doc(source, "go", suffix))
    assert chunk.metadata["symbol"] == "Server, New, Server.Start"


def test_broken_source_falls_back_to_the_text_splitter():
    text = "function broken( {{{\n" + "\n".join(f"  x{i} = {i};" for i in range(80))
    chunks = CodeChunker(chunk_size=500, chunk_overlap=0).split(make_doc(text, "js", ".js"))
    assert len(chunks) > 1
    assert all(c.metadata["symbol"] == "" for c in chunks)


def test_a_few_unparsable_lines_do_not_discard_the_whole_file():
    functions = "".join(f"function f{i}() {{\n  return {i};\n}}\n" for i in range(12))
    text = functions + "const = ;\n" + "function last() {}\n"
    symbols = extract_symbols(text, "js", ".js")
    assert symbols is not None
    assert [s.name for s in symbols][:2] == ["f0", "f1"]
    assert symbols[-1].name == "last"
