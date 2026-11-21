# utils/symbols.py

from dataclasses import dataclass, field

# Loader language name -> tree-sitter grammar name. TSX has its own grammar, picked by file suffix.
GRAMMARS = {
    "js": "javascript",
    "ts": "typescript",
    "java": "java",
    "kotlin": "kotlin",
    "go": "go",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "csharp": "csharp",
    "c": "c",
    "cpp": "cpp",
}

# Node types that define a symbol, and the subset whose body holds more symbols (methods, nested types).
DEFINITIONS = {
    "javascript": {"function_declaration", "generator_function_declaration", "class_declaration", "method_definition"},
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "internal_module",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
    },
    "kotlin": {"class_declaration", "object_declaration", "function_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {
        "function_item",
        "function_signature_item",
        "struct_item",
        "enum_item",
        "union_item",
        "trait_item",
        "impl_item",
        "mod_item",
        "type_item",
    },
    "ruby": {"method", "singleton_method", "class", "module"},
    "php": {
        "function_definition",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
        "method_declaration",
    },
    "csharp": {
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
        "namespace_declaration",
    },
    "c": {"function_definition", "struct_specifier", "enum_specifier", "union_specifier", "type_definition"},
    "cpp": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "union_specifier",
        "class_specifier",
        "type_definition",
        "namespace_definition",
    },
}
DEFINITIONS["tsx"] = DEFINITIONS["typescript"]

CONTAINERS = {
    "class_declaration",
    "abstract_class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "object_declaration",
    "trait_item",
    "trait_declaration",
    "impl_item",
    "mod_item",
    "internal_module",
    "namespace_declaration",
    "namespace_definition",
    "struct_declaration",
    "struct_specifier",
    "class_specifier",
    "class",
    "module",
}

# Nodes that wrap a definition (export, template, extern block) and only add their own lines to it.
WRAPPERS = {
    "export_statement",
    "template_declaration",
    "expression_statement",
    "ambient_declaration",
    "linkage_specification",
}

# Nodes between a container and its members.
BODIES = {
    "class_body",
    "interface_body",
    "enum_body",
    "enum_body_declarations",
    "enum_class_body",
    "declaration_list",
    "field_declaration_list",
    "body_statement",
    "statement_block",
}

NAME_TYPES = {"identifier", "type_identifier", "simple_identifier", "constant", "name", "field_identifier"}
FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
# Lines directly above a definition that belong to it (doc comments, Rust attributes).
ATTACHED = ("comment", "attribute_item")

MAX_ERROR_SHARE = 0.1

# tree-sitter counts rows by \n only; text with other line breaks would drift from str.splitlines().
ODD_BREAKS = tuple(chr(c) for c in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029))


@dataclass
class Symbol:
    name: str
    kind: str  # the tree-sitter node type
    start: int  # 1-based, inclusive; includes comments directly above
    end: int
    children: list["Symbol"] = field(default_factory=list)


def grammar_for(language: str, suffix: str) -> str | None:
    if suffix.lower() == ".tsx":
        return "tsx"
    return GRAMMARS.get(language)


def extract_symbols(text: str, language: str, suffix: str = "") -> list[Symbol] | None:
    """
    Definitions of a source file as a tree (classes hold their methods), with exact line ranges.
    Returns None when the language is not supported or the file does not parse cleanly, so the
    caller can fall back to plain text splitting.
    """
    grammar = grammar_for(language, suffix)
    if grammar is None or "\r" in text.replace("\r\n", "") or any(c in text for c in ODD_BREAKS):
        return None
    parser = _parser(grammar)
    if parser is None:
        return None
    root = parser.parse(text.encode("utf-8")).root_node
    # Macro-heavy C and vendored files often have a few unparsable spots; only a mostly broken file is skipped.
    if root.has_error and _error_lines(root) > MAX_ERROR_SHARE * (root.end_point[0] + 1):
        return None
    return _collect(root, DEFINITIONS[grammar], grammar)


def _error_lines(root) -> int:
    """Number of lines covered by syntax errors."""
    lines: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            lines.update(range(node.start_point[0], node.end_point[0] + 1))
        elif node.has_error:
            stack.extend(node.children)
    return len(lines)


def _parser(grammar: str):
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(grammar)
    except Exception:  # missing grammar download or an unsupported build
        return None


def _collect(parent, defs: set[str], grammar: str) -> list[Symbol]:
    """Symbols among the direct members of a node, looking through bodies and wrappers."""
    symbols: list[Symbol] = []
    children = parent.named_children
    for i, node in enumerate(children):
        if node.type in BODIES or (parent.type != node.type and _is_body_field(parent, node)):
            symbols += _collect(node, defs, grammar)
            continue
        found = _symbols_of(node, defs, grammar)
        if len(found) == 1 and node.type in WRAPPERS | {"lexical_declaration", "variable_declaration"}:
            found[0].start = min(found[0].start, node.start_point[0] + 1)
            found[0].end = max(found[0].end, node.end_point[0] + 1)
        for symbol in found:
            symbol.start = min(symbol.start, _attached_start(children, i))
        symbols += found
    return symbols


def _is_body_field(parent, node) -> bool:
    body = parent.child_by_field_name("body")
    return body is not None and body.id == node.id and node.type != "block"


def _attached_start(children, index: int) -> int:
    """First line of the comments or attributes sitting right above children[index]."""
    start = children[index].start_point[0] + 1
    for prev in reversed(children[:index]):
        if not prev.type.endswith(ATTACHED) and "comment" not in prev.type:
            break
        if prev.end_point[0] + 1 < start - 1:
            break
        start = prev.start_point[0] + 1
    return start


def _symbols_of(node, defs: set[str], grammar: str) -> list[Symbol]:
    if node.type in defs and _is_definition(node):
        name = _name(node)
        symbol = Symbol(name, node.type, node.start_point[0] + 1, node.end_point[0] + 1)
        if node.type in CONTAINERS:
            symbol.children = _collect(node, defs, grammar)
        if node.type == "method_declaration" and grammar == "go":
            symbol.name = _go_method_name(node, name)
        return _split_groups(node, symbol)
    if node.type in WRAPPERS or node.type == "declaration":
        return [s for child in node.named_children for s in _symbols_of(child, defs, grammar)]
    if node.type in {"lexical_declaration", "variable_declaration"}:
        return _function_variables(node)
    if node.type in BODIES:
        return _collect(node, defs, grammar)
    return []


def _split_groups(node, symbol: Symbol) -> list[Symbol]:
    """A Go `type ( A struct{}; B struct{} )` group is one node holding several named types."""
    if node.type != "type_declaration":
        return [symbol]
    specs = [c for c in node.named_children if c.type in {"type_spec", "type_alias"}]
    if len(specs) <= 1:
        return [symbol]
    return [
        Symbol(_text(s.child_by_field_name("name")), "type_spec", s.start_point[0] + 1, s.end_point[0] + 1)
        for s in specs
    ]


def _is_definition(node) -> bool:
    # `struct s x;` mentions a type without defining it
    if node.type in {"struct_specifier", "enum_specifier", "union_specifier", "class_specifier"}:
        return node.child_by_field_name("body") is not None
    if node.type == "type_definition":
        return any(c.type.endswith("_specifier") and c.child_by_field_name("body") for c in node.named_children)
    return True


def _function_variables(node) -> list[Symbol]:
    """`const run = () => ...` defines a function called run."""
    declarators = [c for c in node.named_children if c.type == "variable_declarator"]
    symbols = []
    for declarator in declarators:
        value = declarator.child_by_field_name("value")
        name = declarator.child_by_field_name("name")
        if value is not None and name is not None and value.type in FUNCTION_VALUES:
            symbols.append(Symbol(_text(name), "function", node.start_point[0] + 1, node.end_point[0] + 1))
    return symbols[:1] if len(declarators) == 1 else symbols


def _name(node) -> str:
    named = node.child_by_field_name("name")
    if named is not None:
        return _text(named)
    kind = node.type
    if kind == "type_declaration":
        spec = next((c for c in node.named_children if c.type in {"type_spec", "type_alias"}), None)
        return _text(spec.child_by_field_name("name")) if spec is not None else ""
    if kind == "impl_item":
        return _text(node.child_by_field_name("type"))
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        return _declarator_name(declarator)
    # Kotlin and others keep the name as a plain child
    for child in node.named_children:
        if child.type in NAME_TYPES:
            return _text(child)
    return "anonymous" if kind.startswith("namespace") else ""


def _declarator_name(node) -> str:
    """C and C++ wrap the name of a function in pointer and function declarators."""
    while node is not None and node.type not in NAME_TYPES | {
        "qualified_identifier",
        "destructor_name",
        "operator_name",
    }:
        node = node.child_by_field_name("declarator")
    return _text(node)


def _go_method_name(node, name: str) -> str:
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return name
    for child in receiver.named_children:
        type_node = child.child_by_field_name("type")
        if type_node is not None:
            return f"{_text(type_node).lstrip('*')}.{name}"
    return name


def _text(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""
