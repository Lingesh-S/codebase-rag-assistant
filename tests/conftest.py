import hashlib
import math
import textwrap
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from rag_pipeline.pipeline import RAGPipeline, RAGSettings
from retriever.code_retriever import tokenize_code


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings so tests run offline and fast."""

    def __init__(self, dim: int = 512, model_name: str = "fake-hash-embeddings"):
        self.dim = dim
        self.model_name = model_name

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        vector[0] = 0.01  # keeps empty texts away from the zero vector
        for token in tokenize_code(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeLLM:
    """Records every request and replies with a canned answer."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "It hashes passwords with PBKDF2 [1]."):
        self.reply = reply
        self.calls: list[dict] = []

    def stream(self, system, messages, max_tokens=None, effort=None):
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens, "effort": effort})
        for i in range(0, len(self.reply), 4):
            yield self.reply[i:i + 4]


def write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


@pytest.fixture
def codebase(tmp_path) -> Path:
    root = tmp_path / "project"
    write(root, "app/auth.py", '''
        """Password helpers."""
        import hashlib


        def hash_password(password: str, salt: bytes) -> str:
            """Hashes a password with PBKDF2."""
            return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 1000).hex()
    ''')
    write(root, "app/cart.py", '''
        class ShoppingCart:
            """Holds items before checkout."""

            def __init__(self):
                self.items = []

            def add_item(self, sku, quantity):
                self.items.append((sku, quantity))

            def total_quantity(self):
                return sum(q for _, q in self.items)
    ''')
    write(root, "web/client.js", '''
        export function fetchWithRetry(url, retries = 3) {
          return fetch(url).catch(() => (retries > 0 ? fetchWithRetry(url, retries - 1) : null));
        }
    ''')
    write(root, "README.md", "# Demo shop\n\nA tiny shop used in tests.\n")
    write(root, "settings.json", '{"currency": "EUR", "max_cart_items": 50}\n')
    # Everything below must be ignored by the loader.
    write(root, "node_modules/lib/index.js", "module.exports = 1;\n")
    write(root, "venv/lib/site.py", "print('venv')\n")
    write(root, ".git/config.json", '{"core": true}\n')
    write(root, ".env", "API_KEY=super-secret\n")
    write(root, "secrets.toml", 'token = "super-secret"\n')
    write(root, "image.py", b"\x89PNG\x00\x00binary")
    write(root, "empty.py", "\n\n")
    return root


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def make_rag(tmp_path, fake_embeddings):
    def factory(root: Path, reranker=None, **settings) -> RAGPipeline:
        settings.setdefault("persist_dir", str(tmp_path / "chroma"))
        settings.setdefault("rerank_model", None)
        return RAGPipeline(root, RAGSettings(**settings), collection_name="test-code", embedding=fake_embeddings, reranker=reranker)

    return factory
