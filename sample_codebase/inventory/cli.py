# sample_codebase/inventory/cli.py

"""
Command line interface for the inventory service.
"""

import argparse
import json

from inventory.service import InventoryService
from inventory.storage import JsonInventoryRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inventory", description="Manage warehouse stock")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="receive stock for a SKU")
    add.add_argument("sku")
    add.add_argument("amount", type=int)

    remove = sub.add_parser("remove", help="ship stock for a SKU")
    remove.add_argument("sku")
    remove.add_argument("amount", type=int)

    sub.add_parser("report", help="print low stock products and total value")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with open("config.json", encoding="utf-8") as f:
        config = json.load(f)
    service = InventoryService(JsonInventoryRepository(config["data_file"]))

    if args.command == "add":
        product = service.add_stock(args.sku, args.amount)
        print(f"{product.sku}: {product.quantity} in stock")
    elif args.command == "remove":
        product = service.remove_stock(args.sku, args.amount)
        print(f"{product.sku}: {product.quantity} in stock")
    else:
        for product in service.low_stock_report():
            print(f"LOW  {product.sku:<10} {product.quantity:>5}")
        print(f"Total value: {service.total_inventory_value():.2f} {config['currency']}")


if __name__ == "__main__":
    main()
