# sample_codebase/inventory/service.py

"""
Business rules for stock management.
"""

from inventory.models import MovementType, Product, StockMovement
from inventory.storage import JsonInventoryRepository


class InsufficientStockError(Exception):
    """Raised when a shipment asks for more units than are in stock."""


class UnknownProductError(KeyError):
    """Raised when an operation references a SKU that does not exist."""


class InventoryService:
    def __init__(self, repository: JsonInventoryRepository):
        self.repository = repository
        self.movements: list[StockMovement] = []

    def add_stock(self, sku: str, amount: int) -> Product:
        """Receives new units of a product and records the movement."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        product = self._require(sku)
        product.quantity += amount
        self._record(sku, amount, MovementType.RECEIVED)
        return product

    def remove_stock(self, sku: str, amount: int) -> Product:
        """
        Ships units of a product.

        Raises InsufficientStockError instead of letting the quantity go
        negative, so the stored inventory never drops below zero.
        """
        product = self._require(sku)
        if amount > product.quantity:
            raise InsufficientStockError(
                f"Cannot remove {amount} of {sku}: only {product.quantity} left"
            )
        product.quantity -= amount
        self._record(sku, -amount, MovementType.SHIPPED)
        return product

    def low_stock_report(self) -> list[Product]:
        """Returns products that need to be reordered, lowest quantity first."""
        low = [p for p in self.repository.all() if p.needs_reorder()]
        return sorted(low, key=lambda p: p.quantity)

    def total_inventory_value(self) -> float:
        """Sums price times quantity across all products and returns dollars."""
        cents = sum(p.value_cents() for p in self.repository.all())
        return cents / 100

    def _require(self, sku: str) -> Product:
        product = self.repository.get(sku)
        if product is None:
            raise UnknownProductError(sku)
        return product

    def _record(self, sku: str, change: int, movement_type: MovementType) -> None:
        self.movements.append(StockMovement(sku=sku, change=change, movement_type=movement_type))
        self.repository.save()
