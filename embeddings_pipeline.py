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

Schema
--------------------------------------
TABLE products(
  product_id INTEGER PRIMARY KEY,
  parent_asin TEXT,
  main_category TEXT,
  title TEXT,
  features_text TEXT,
  details_text TEXT
);
TABLE reviews(
  parent_asin TEXT,
  title TEXT,
  text TEXT,
  helpful_vote INTEGER
);

"""
from __future__ import annotations
import os
import sys
import io
import math
import json
import time
import argparse
import sqlite3
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# -----------------------------
# Utils
# -----------------------------

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, ord=2)
    if norm < eps:
        return v
    return v / norm


def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attn_mask: [B, T]
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-6)
    return summed / count


def device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        # MPS autocast is still finicky; keep float32
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


# -----------------------------
# Database access
# -----------------------------

def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def count_products(conn: sqlite3.Connection, categories: Optional[List[str]]) -> int:
    cur = conn.cursor()
    if categories:
        q = f"SELECT COUNT(*) FROM products WHERE main_category IN ({','.join(['?']*len(categories))})"
        cur.execute(q, categories)
    else:
        cur.execute("SELECT COUNT(*) FROM products")
    return int(cur.fetchone()[0])


def fetch_products_page(
    conn: sqlite3.Connection,
    start: int,
    limit: int,
    categories: Optional[List[str]]
) -> List[sqlite3.Row]:
    cur = conn.cursor()
    if categories:
        q = (
            "SELECT product_id, parent_asin, main_category, product_title AS title, features_json AS features_text, details_json AS details_text\n"
            "FROM products\n"
            f"WHERE main_category IN ({','.join(['?']*len(categories))})\n"
            "ORDER BY product_id ASC\n"
            "LIMIT ? OFFSET ?"
        )
        cur.execute(q, (*categories, limit, start))
    else:
        q = (
            "SELECT product_id, parent_asin, main_category, product_title AS title, features_json AS features_text, details_json AS details_text\n"
            "FROM products\n"
            "ORDER BY product_id ASC\n"
            "LIMIT ? OFFSET ?"
        )
        cur.execute(q, (limit, start))
    rows = cur.fetchall()
    return rows


def fetch_topk_reviews(
    conn: sqlite3.Connection, parent_asin: str, k: int
) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(review_title, '') AS title, COALESCE(review_text, '') AS text
        FROM reviews
        WHERE parent_asin = ?
        ORDER BY helpful_vote DESC
        LIMIT ?
        """,
        (parent_asin, k),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


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
    if features_text:
        parts.append(str(features_text))
    if details_text:
        parts.append(str(details_text))
    # Append reviews
    for (rt, rx) in reviews:
        if rt:
            parts.append(str(rt))
        if rx:
            parts.append(str(rx))
    doc = "\n".join(p.strip() for p in parts if p and p.strip())
    if len(doc) < min_chars:
        return None
    return doc


def chunk_with_tokenizer(
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    stride: int,
) -> Dict[str, List[torch.Tensor]]:
    # Use HF overflow + stride to get overlapping chunks
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors=None,
        padding=False,
    )
    # Ensure list-of-chunks format
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------
# Embedding model wrapper
# -----------------------------
class BlairEncoder(nn.Module):
    def __init__(self, model_name: str = "hyp1231/blair-roberta-large"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def encode_batches(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: torch.device,
        dtype: torch.dtype,
        max_length: int,
        stride: int,
        batch_size: int,
    ) -> List[np.ndarray]:
        """Return 1 vector per input text (avg of chunk embeddings, L2-normalized)."""
        out_vecs: List[np.ndarray] = []
        use_autocast = (device.type == "cuda")

        for text in texts:
            # Chunk into token windows
            chunks = chunk_with_tokenizer(tokenizer, text, max_length, stride)
            if len(chunks["input_ids"]) == 0:
                out_vecs.append(np.zeros(self.model.config.hidden_size, dtype=np.float32))
                continue

            # Embed chunks in batches
            chunk_vecs: List[torch.Tensor] = []
            # Build simple index slicing over chunks
            ids_list = chunks["input_ids"]
            attn_list = chunks["attention_mask"]
            total = len(ids_list)
            for i in range(0, total, batch_size):
                batch_ids = ids_list[i:i+batch_size]
                batch_attn = attn_list[i:i+batch_size]
                # Pad to max length within batch
                enc = tokenizer.pad(
                    {"input_ids": batch_ids, "attention_mask": batch_attn},
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                    outputs = self.model(**enc)
                    last_hidden = outputs.last_hidden_state  # [B, T, H]
                    pooled = mean_pool(last_hidden, enc["attention_mask"])  # [B, H]
                chunk_vecs.append(pooled.detach())

            # Average across chunks -> numpy
            all_chunks = torch.cat(chunk_vecs, dim=0)
            prod_vec = all_chunks.mean(dim=0)
            prod_vec = torch.nn.functional.normalize(prod_vec, p=2, dim=0)
            out_vecs.append(prod_vec.cpu().float().numpy())

        return out_vecs


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    device, dtype = device_and_dtype()
    print(f"Using device={device}, dtype={dtype}")

    print("Loading tokenizer/model…")
    tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-large")
    encoder = BlairEncoder("hyp1231/blair-roberta-large").to(device)
    encoder.eval()

    conn = open_db(args.db)
    total = count_products(conn, args.categories)
    print(f"Total products matching filter: {total}")

    start = args.offset
    shard_idx = 0
    pbar = tqdm(total=max(0, total - start), desc="Embedding products")

    while start < total:
        rows = fetch_products_page(conn, start, args.page_size, args.categories)
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
            rv = fetch_topk_reviews(conn, asin, args.topk_reviews) if asin else []
            doc = build_product_document(title, features, details, rv, min_chars=args.min_chars)
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
        vecs = encoder.encode_batches(
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            dtype=dtype,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.batch_size,
        )

        # Post L2 (safety) + pack to float32
        out_ids = np.asarray(ids, dtype=np.int64)
        out_asins = np.asarray(asins)
        out_vecs = np.stack([l2_normalize(v.astype(np.float32)) for v in vecs], axis=0)

        shard_name = f"shard_{start}_{start+len(rows)-1}_{shard_idx:05d}.npz"
        shard_path = os.path.join(args.out_dir, shard_name)
        np.savez_compressed(shard_path, ids=out_ids, asins=out_asins, vectors=out_vecs)

        shard_idx += 1
        start += len(rows)
        pbar.update(len(rows))

    pbar.close()
    conn.close()
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
        if not fn.endswith('.npz'):
            continue
        data = np.load(os.path.join(shards_dir, fn))
        ids = data['ids']
        asins = data['asins']
        vecs = data['vectors']
        if dim is None:
            dim = vecs.shape[1]
        dfs.append(pd.DataFrame({
            'product_id': ids,
            'parent_asin': asins,
            'vector': list(vecs),
        }))
    if not dfs:
        print("No shards found.")
        return
    df = pd.concat(dfs, ignore_index=True)

    # Build Arrow schema with FixedSizeList for vector
    vector_values = np.concatenate(df['vector'].values).astype(np.float32)
    arr_values = pa.array(vector_values)
    arr_list = pa.FixedSizeListArray.from_arrays(arr_values, list_size=dim)
    table = pa.Table.from_arrays([
        pa.array(df['product_id'].values),
        pa.array(df['parent_asin'].values),
        arr_list
    ], names=['product_id', 'parent_asin', 'vector'])

    pq.write_table(table, out_parquet)
    print(f"Wrote {out_parquet} (rows={len(df)}, dim={dim})")


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CS6400 Embedding Pipeline — BLAIR")
    p.add_argument('--db', required=True, help='Path to SQLite database file')
    p.add_argument('--out_dir', required=True, help='Directory to write .npz shards')
    p.add_argument('--categories', nargs='*', default=None, help='Optional main_category filter list')

    p.add_argument('--page_size', type=int, default=512, help='Products fetched per page')
    p.add_argument('--offset', type=int, default=0, help='Starting row offset')

    p.add_argument('--batch_size', type=int, default=64, help='Chunk batch size for model forward')
    p.add_argument('--max_length', type=int, default=64, help='Max tokens per chunk (proposal: 64)')
    p.add_argument('--stride', type=int, default=10, help='Token overlap between chunks')

    p.add_argument('--topk_reviews', type=int, default=20, help='Max number of reviews to append per product')
    p.add_argument('--min_chars', type=int, default=20, help='Drop products with document shorter than this')

    p.add_argument('--parquet', type=str, default=None, help='If set, convert shards to this Parquet file at end')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
    if args.parquet:
        shards_to_parquet(args.out_dir, args.parquet)
