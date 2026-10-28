# Sample Inventory Service

A small demo project used to try out the Codebase RAG Assistant and to
evaluate retrieval quality.

It tracks products in a warehouse:

- `inventory/models.py` defines the `Product` and `StockMovement` data types.
- `inventory/storage.py` persists products to a JSON file.
- `inventory/service.py` holds the business rules for adding and removing stock,
  low-stock reports and inventory valuation.
- `inventory/auth.py` hashes user passwords and issues API tokens.
- `inventory/cli.py` exposes the service as a command line tool.
- `web/api_client.js` is a browser client for the HTTP API.
- `config.json` holds runtime settings such as the data file location.

Run the CLI with:

```bash
python -m inventory.cli report
```
