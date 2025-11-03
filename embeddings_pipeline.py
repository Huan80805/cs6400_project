#!/usr/bin/env python3
"""
Implements the embedding plan from the project proposal:
• Model: hyp1231/blair-roberta-large (BLAIR)
• Text aggregation: product text + (top-k) review snippets
• Token-chunking with overlap (default: max_length=64, stride=10)
• Per-chunk embeddings via mean pooling
• Vector aggregation across chunks by element-wise average
• Final L2 normalization
• Output written as sharded .npz files (ids, asins, vectors)

Quick start to run
-----------
python embeddings_pipeline.py \
  --db /path/to/amazon.sqlite \
  --out_dir ./emb_shards \
  --batch_size 64 \
  --page_size 512 \
  --max_length 64 \
  --stride 10 \
  --topk_reviews 20 \
  --min_chars 20 \
  --categories All_Beauty Amazon_Fashion Appliances

Dependencies
------------
python -m pip install torch transformers numpy tqdm
# optional (if you later want to convert shards to Parquet)
python -m pip install pandas pyarrow

"""

from __future__ import annotations
import os
import argparse
from typing import List, Tuple, Optional
from encoder import Encoder, EncoderConfig
from db import DB

import numpy as np
from tqdm import tqdm

# -----------------------------
# Text building and chunking
# -----------------------------


def build_product_document(
    title: Optional[str],
    features_text: Optional[str],
    details_text: Optional[str],
    reviews: List[Tuple[str, str]],
    min_chars: int = 0,
) -> Optional[str]:
    parts: List[str] = []
    if title:
        parts.append(str(title))
    # Commented out to focus on only title, but even so the recall is bad
    # if features_text:
    #     parts.append(str(features_text))
    # if details_text:
    #     parts.append(str(details_text))
    # # Append reviews
    # for rt, rx in reviews:
    #     if rt:
    #         parts.append(str(rt))
    #     if rx:
    #         parts.append(str(rx))
    doc = "\n".join(p.strip() for p in parts if p and p.strip())
    if len(doc) < min_chars:
        return None
    return doc


# -----------------------------
# Main pipeline
# -----------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    db = DB(args.db)
    total = db.count_products(args.categories)
    print(f"Total products matching filter: {total}")
    print("Loading tokenizer/model…")
    encoder = Encoder(
        EncoderConfig(
            model_name="hyp1231/blair-roberta-large",
        )
    )

    start = args.offset
    shard_idx = 0
    pbar = tqdm(total=max(0, total - start), desc="Embedding products")

    while start < total:
        rows = db.fetch_products_page(start, args.page_size, args.categories)
        if not rows:
            break

        ids: List[int] = []
        asins: List[str] = []
        texts: List[str] = []

        for r in rows:
            pid = int(r["product_id"]) if r["product_id"] is not None else None
            asin = r["parent_asin"] if r["parent_asin"] is not None else ""
            title = r["title"]
            features = r["features_text"]
            details = r["details_text"]

            # pull top-k reviews to augment text
            rv = db.fetch_topk_reviews(asin, args.topk_reviews) if asin else []
            doc = build_product_document(
                title, features, details, rv, min_chars=args.min_chars
            )
            if doc is None:
                # skip if too short
                continue
            ids.append(pid)
            asins.append(asin)
            texts.append(doc)

        if not texts:
            # advance and continue
            start += len(rows)
            pbar.update(len(rows))
            continue

        # Encode (1 vector per product)
        out_vecs = encoder.encode_documents_in_batches(
            texts=texts,
            max_length=args.max_length,
            stride=args.stride,
        )

        out_ids = np.asarray(ids, dtype=np.int64)
        out_asins = np.asarray(asins)

        shard_name = f"shard_{start}_{start + len(rows) - 1}_{shard_idx:05d}.npz"
        shard_path = os.path.join(args.out_dir, shard_name)
        np.savez_compressed(shard_path, ids=out_ids, asins=out_asins, vectors=out_vecs)

        shard_idx += 1
        start += len(rows)
        pbar.update(len(rows))

    pbar.close()
    db.close()
    print("Done.")


# -----------------------------
# convert shards -> Parquet (one file)
# -----------------------------


def shards_to_parquet(shards_dir: str, out_parquet: str) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Read shards into a table with a fixed-size list column for vectors
    dfs = []
    dim = None
    for fn in sorted(os.listdir(shards_dir)):
        if not fn.endswith(".npz"):
            continue
        data = np.load(os.path.join(shards_dir, fn))
        ids = data["ids"]
        asins = data["asins"]
        vecs = data["vectors"]
        if dim is None:
            dim = vecs.shape[1]
        dfs.append(
            pd.DataFrame(
                {
                    "product_id": ids,
                    "parent_asin": asins,
                    "vector": list(vecs),
                }
            )
        )
    if not dfs:
        print("No shards found.")
        return
    df = pd.concat(dfs, ignore_index=True)

    # Build Arrow schema with FixedSizeList for vector
    vector_values = np.concatenate(df["vector"].values).astype(np.float32)
    arr_values = pa.array(vector_values)
    arr_list = pa.FixedSizeListArray.from_arrays(arr_values, list_size=dim)
    table = pa.Table.from_arrays(
        [
            pa.array(df["product_id"].values),
            pa.array(df["parent_asin"].values),
            arr_list,
        ],
        names=["product_id", "parent_asin", "vector"],
    )

    pq.write_table(table, out_parquet)
    print(f"Wrote {out_parquet} (rows={len(df)}, dim={dim})")


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CS6400 Embedding Pipeline — BLAIR")
    p.add_argument("--db", required=True, help="Path to SQLite database file")
    p.add_argument("--out_dir", required=True, help="Directory to write .npz shards")
    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional main_category filter list",
    )

    p.add_argument(
        "--page_size", type=int, default=1024, help="Products fetched per page"
    )
    p.add_argument("--offset", type=int, default=0, help="Starting row offset")

    p.add_argument(
        "--batch_size", type=int, default=64, help="Chunk batch size for model forward"
    )
    p.add_argument(
        "--max_length", type=int, default=64, help="Max tokens per chunk (proposal: 64)"
    )
    p.add_argument(
        "--stride", type=int, default=10, help="Token overlap between chunks"
    )

    p.add_argument(
        "--topk_reviews",
        type=int,
        default=5,
        help="Max number of reviews to append per product",
    )
    p.add_argument(
        "--min_chars",
        type=int,
        default=20,
        help="Drop products with document shorter than this",
    )

    p.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="If set, convert shards to this Parquet file at end",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
    if args.parquet:
        shards_to_parquet(args.out_dir, args.parquet)
