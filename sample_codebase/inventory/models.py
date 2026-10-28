# sample_codebase/inventory/models.py

"""
Data types for the inventory service.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class MovementType(Enum):
    """Direction of a stock change."""

    RECEIVED = "received"
    SHIPPED = "shipped"
    ADJUSTED = "adjusted"


@dataclass
class Product:
    """
    A product stored in the warehouse.

    Prices are stored as integer cents to avoid floating point rounding errors.
    """

    sku: str
    name: str
    price_cents: int
    quantity: int = 0
    reorder_level: int = 5

    def needs_reorder(self) -> bool:
        """Returns True when the quantity is at or below the reorder level."""
        return self.quantity <= self.reorder_level

    def value_cents(self) -> int:
        """Returns the total value of this product's stock in cents."""
        return self.price_cents * self.quantity


@dataclass
class StockMovement:
    """An audit record of a single stock change."""

    sku: str
    change: int
    movement_type: MovementType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
