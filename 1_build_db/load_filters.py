#!/usr/bin/env python3
"""
Reads a JSON file containing product filters and inserts into the
specified filter table in the SQLite database.

The JSON file is expected to be a list of objects, where each object
has at least a "parent_asin" and a "filters" key.

[
    {
       "parent_asin": "B00XXXXX",
       "filters": [{"filter_column": "main_category",...}]
    },
]

Usage:
    # For ESCI filters
    python load_filters.py --db amz.db --json esci_filters.json --table esci_filters
    
    # For Amazon-C4 filters
    python load_filters.py --db amz.db --json amz_c4_filters.json --table amz_c4_filters
"""

import sqlite3
import json
import argparse
import sys
from tqdm import tqdm

# Valid filter table names
VALID_TABLES = {"esci_filters", "amz_c4_filters"}


def load_filters_to_db(db_path: str, json_path: str, table_name: str, batch_size: int = 5000):
    """
    Connects to the DB and inserts filter data into the specified table.
    """
    # Validate table name to prevent SQL injection
    if table_name not in VALID_TABLES:
        print(f"Error: Invalid table name '{table_name}'. Must be one of: {VALID_TABLES}", file=sys.stderr)
        return
    
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

    update_params = []
    for item in data:
        if "parent_asin" not in item or "filters" not in item:
            print(
                f"Warning: Skipping item without 'parent_asin' or 'filters': {item}",
                file=sys.stderr,
            )
            continue

        try:
            parent_asin = str(item["parent_asin"])  # Ensure ASIN is string
            filters_list = item["filters"]
            # Convert the list of filters to a single JSON string
            filters_json = json.dumps(filters_list)
            update_params.append((parent_asin, filters_json))
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

        # Clear existing filters in the target table
        print(f"Clearing existing {table_name} table...")
        cur.execute(f"DELETE FROM {table_name}")
        conn.commit()

        # Get mapping from parent_asin to product_id
        print("Fetching product_id mapping...")
        cur.execute("SELECT parent_asin, product_id FROM products")
        asin_to_id = {row[0]: row[1] for row in cur.fetchall()}
        print(f"Loaded {len(asin_to_id)} product mappings.")

        # Prepare data with correct product_id
        final_params = []
        skipped_count = 0
        for asin, filters_json in update_params:
            # asin is already a string from the JSON parsing loop below
            if asin in asin_to_id:
                final_params.append((asin_to_id[asin], filters_json))
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Warning: {skipped_count} filters skipped because ASIN not found in products table.")

        # insert into the specified filter table
        update_query = f"""
            INSERT INTO {table_name} (product_id, filters_json)
            VALUES (?, ?)
        """

        # Use executemany in batches
        for i in tqdm(
            range(0, len(final_params), batch_size), desc="Updating DB batches"
        ):
            batch = final_params[i : i + batch_size]
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
        "--table",
        required=True,
        choices=["esci_filters", "amz_c4_filters"],
        help="Target filter table name",
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
    load_filters_to_db(args.db, args.json, args.table, args.batch_size)
