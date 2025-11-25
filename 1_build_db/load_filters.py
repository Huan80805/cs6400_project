#!/usr/bin/env python3
"""
Reads a JSON file containing product filters and updates the 'products'
table in the SQLite database.

The JSON file is expected to be a list of objects, where each object
has at least a "product_id" and a "filters" key.

[
    {
       "product_id": 123,
       "filters": [{"filter_column": "main_category",...}]
    },
]

Usage:
    python load_filters.py --db amz.db --json filters_deduplicated.json
"""

import sqlite3
import json
import argparse
import sys
from tqdm import tqdm


def load_filters_to_db(db_path: str, json_path: str, batch_size: int = 5000):
    """
    Connects to the DB and updates product rows with filter data.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            print(f"Loading data from {json_path}...")
            data = json.load(f)
            if not isinstance(data, list):
                print(
                    f"Error: Expected JSON file to contain a list of objects.",
                    file=sys.stderr,
                )
                return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}", file=sys.stderr)
        return
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading file {json_path}: {e}", file=sys.stderr)
        return

    print(f"Loaded {len(data)} product filter entries.")

    # Prepare data for executemany
    # (json_string, product_id)
    update_params = []
    for item in data:
        if "product_id" not in item or "filters" not in item:
            print(
                f"Warning: Skipping item without 'product_id' or 'filters': {item}",
                file=sys.stderr,
            )
            continue

        try:
            product_id = int(item["product_id"])
            filters_list = item["filters"]
            # Convert the list of filters to a single JSON string
            filters_json = json.dumps(filters_list)
            update_params.append((filters_json, product_id))
        except Exception as e:
            print(
                f"Warning: Skipping item due to processing error: {e} | Item: {item}",
                file=sys.stderr,
            )

    if not update_params:
        print("No valid filter data to update.")
        return

    # Connect to DB and update in batches
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        print(f"Updating {len(update_params)} rows in the database...")

        # Use PRAGMAs for faster inserts
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        update_query = """
            UPDATE products
            SET filters_json = ?
            WHERE product_id = ?
        """

        # Use executemany in batches
        for i in tqdm(
            range(0, len(update_params), batch_size), desc="Updating DB batches"
        ):
            batch = update_params[i : i + batch_size]
            cur.executemany(update_query, batch)
            conn.commit()

        print(
            f"Database update complete. {cur.rowcount} rows affected (note: this may be total across all batches)."
        )

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load product filters from JSON into SQLite DB."
    )
    parser.add_argument(
        "--db", required=True, help="Path to the SQLite database file (e.g., amz.db)"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the filters JSON file (e.g., filters.json)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Number of rows to update per transaction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_filters_to_db(args.db, args.json, args.batch_size)
