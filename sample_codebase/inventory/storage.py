# sample_codebase/inventory/storage.py

"""
JSON file persistence for products.
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from inventory.models import Product


class JsonInventoryRepository:
    """Loads and saves products to a single JSON file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._products: dict[str, Product] = {}
        self.load()

    def load(self) -> None:
        """Reads products from disk. A missing file means an empty inventory."""
        if not self.path.exists():
            self._products = {}
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._products = {item["sku"]: Product(**item) for item in raw}

    def save(self) -> None:
        """
        Writes products to disk atomically.

        The data is written to a temporary file in the same directory first and
        then moved over the real file with os.replace, so a crash never leaves a
        half-written inventory file behind.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(p) for p in self._products.values()]
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.path)

    def get(self, sku: str) -> Product | None:
        return self._products.get(sku)

    def upsert(self, product: Product) -> None:
        self._products[product.sku] = product

    def delete(self, sku: str) -> None:
        self._products.pop(sku, None)

    def all(self) -> list[Product]:
        return list(self._products.values())
