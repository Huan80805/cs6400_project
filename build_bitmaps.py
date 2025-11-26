#!/usr/bin/env python3
import json
import argparse
import pickle
from collections import defaultdict
from typing import Any, Dict, List

from pyroaring import BitMap
from bitmap_keys import make_key   # <- shared key builder

def build_bitmaps_from_filters_json(filters_json_path: str, out_path: str) -> None:
    """
    Build roaring bitmaps from esci_filters_deduplicated.json.

    For each product:
      - product["filters"] is a list of filter specs
      - Each spec has filter_column, filter_value, match_percentage, etc.

    We create a bitmap per (filter_column, op, normalized(filter_value)).
    """
    print(
        f"Building Roaring bitmaps from filters JSON: "
        f"{filters_json_path} → {out_path}"
    )

    with open(filters_json_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    bitmaps: Dict[str, BitMap] = defaultdict(BitMap)

    num_products = 0
    num_specs = 0

    for prod in products:
        num_products += 1
        pid = prod["product_id"]
        filters: List[Dict[str, Any]] = prod.get("filters", [])
        for spec in filters:
            num_specs += 1
            col = spec["filter_column"]
            val = spec["filter_value"]

            # Infer operator like before
            if isinstance(val, list) and len(val) == 2 and all(
                isinstance(x, (int, float)) for x in val
            ):
                op = "BETWEEN"
            elif col in ("features_json", "details_json", "product_title"):
                # text-ish fields we treat as substring matches
                op = "LIKE"
            else:
                op = "="

            key = make_key(col, op, val)
            bitmaps[key].add(pid)

    print(f"Processed {num_products} products.")
    print(f"Processed {num_specs} filter specs.")
    print(f"Built {len(bitmaps)} distinct bitmap keys.")

    with open(out_path, "wb") as f:
        pickle.dump(bitmaps, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved bitmaps to {out_path}")

def main():
    ap = argparse.ArgumentParser()
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

    build_bitmaps_from_filters_json(args.filters_json, args.out)

if __name__ == "__main__":
    main()
