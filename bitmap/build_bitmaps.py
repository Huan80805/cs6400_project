#!/usr/bin/env python3
import json
import argparse
import pickle
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pyroaring import BitMap
from .bitmap_keys import make_key  # shared key builder


def infer_op(col: str, val: Any) -> str:
    """
    Infer the operator given (column, filter_value) from ESCI-style filters.

    - [low, high] numeric  -> BETWEEN
    - text-ish fields      -> LIKE
    - everything else      -> '='
    """
    if isinstance(val, list) and len(val) == 2 and all(
        isinstance(x, (int, float)) for x in val
    ):
        return "BETWEEN"

    # Text-ish fields we treat as substring matches
    if col in ("features_json", "details_json", "product_title"):
        return "LIKE"

    return "="


def load_unique_predicates(filters_json_path: str) -> Dict[str, Tuple[str, str, Any]]:
    """
    Read esci_filters_deduplicated.json and extract *unique* predicates.

    Returns:
        key -> (col, op, val_for_sql)

    where:
      - key is bitmap_keys.make_key(col, op, val_for_key)
      - val_for_sql is what we'll plug into SQL (e.g. raw substring for LIKE,
        (low, high) for BETWEEN, exact value for '=')
    """
    print(f"Loading filter specs from {filters_json_path}...")
    with open(filters_json_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    predicates: Dict[str, Tuple[str, str, Any]] = {}
    num_products = 0
    num_specs = 0

    for prod in products:
        num_products += 1
        filters: List[Dict[str, Any]] = prod.get("filters", [])
        for spec in filters:
            num_specs += 1
            col = spec["filter_column"]
            raw_val = spec["filter_value"]
            op = infer_op(col, raw_val)

            # For key construction we pass raw_val; bitmap_keys.make_key()
            # is responsible for normalizing LIKE values (e.g. stripping '%').
            key = make_key(col, op, raw_val)

            # For SQL:
            #   - BETWEEN: (low, high)
            #   - LIKE: raw substring (we add '%'..'%' in SQL)
            #   - '=': exact value
            if key not in predicates:
                predicates[key] = (col, op, raw_val)

    print(f"  Loaded {num_products} products from JSON.")
    print(f"  Saw {num_specs} filter specs.")
    print(f"  Unique predicates (keys): {len(predicates)}")
    return predicates


def build_bitmaps_full_db(
    db_path: str,
    filters_json_path: str,
    out_path: str,
) -> None:
    """
    Build Roaring bitmaps over the **full products table**.

    Steps:
      1. Load unique predicates from ESCI filters JSON.
      2. For each predicate, run an SQL query against the products table
         to find all product_id that satisfy it.
      3. Store product_ids in BitMaps keyed by make_key(col, op, val).
    """
    print(
        f"Building Roaring bitmaps from full DB {db_path} "
        f"using filters in {filters_json_path} → {out_path}"
    )

    predicates = load_unique_predicates(filters_json_path)

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    bitmaps: Dict[str, BitMap] = defaultdict(BitMap)

    total_predicates = len(predicates)
    print(f"Scanning DB for {total_predicates} predicates...")

    for i, (key, (col, op, val)) in enumerate(predicates.items(), start=1):
        if i % 50 == 0 or i == 1 or i == total_predicates:
            print(f"  [{i}/{total_predicates}] Building bitmap for key: {key}")

        # Build SQL and parameters based on operator
        if op == "BETWEEN":
            if not (isinstance(val, list) or isinstance(val, tuple)) or len(val) != 2:
                # Malformed; skip
                continue
            low, high = val
            sql = f"SELECT product_id FROM products WHERE {col} BETWEEN ? AND ?"
            params = (low, high)
        elif op == "LIKE":
            # val is raw substring; we wrap in '%'..'%' for SQL LIKE
            sql = f"SELECT product_id FROM products WHERE {col} LIKE '%' || ? || '%'"
            params = (val,)
        else:  # '='
            sql = f"SELECT product_id FROM products WHERE {col} = ?"
            params = (val,)

        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.OperationalError as e:
            print(f"    [WARN] SQL error for key={key}: {e}")
            continue

        bm = bitmaps[key]
        for (pid,) in rows:
            # product_id is INTEGER PRIMARY KEY in products
            bm.add(int(pid))

    conn.close()

    # Some summary stats
    num_keys = len(bitmaps)
    total_ids = sum(len(bm) for bm in bitmaps.values())
    print(f"Finished building bitmaps.")
    print(f"  Non-empty bitmap keys: {num_keys}")
    print(f"  Total (key, product_id) memberships: {total_ids}")

    with open(out_path, "wb") as f:
        pickle.dump(bitmaps, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved bitmaps to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        required=True,
        help="Path to amz.db (SQLite file with products table)",
    )
    ap.add_argument(
        "--filters_json",
        required=True,
        help="Path to esci_filters_deduplicated.json",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output path for pickle (bitmaps.pkl)",
    )
    args = ap.parse_args()

    build_bitmaps_full_db(args.db, args.filters_json, args.out)


if __name__ == "__main__":
    main()
