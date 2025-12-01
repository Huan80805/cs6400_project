import numpy as np
import time
import sys
import os
import argparse
from tqdm import tqdm
from typing import Any, Optional, Callable, Dict, List

# New imports for optimized Parquet I/O
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Optional, Literal
from index import FlatL2, IVFPQ

# Import our classes
from db import DB
from encoder import Encoder
from vector_store import VectorStore
from bitmap.roaring_index import RoaringIndex

from search import (
    Search,
    FlatPostfilter,
    FlatPrefilter,
    FlatPostfilterRoaring,
    FlatPrefilterRoaring,
    IVFPQPostfilter,
    IVFPQPostfilterRoaring,
    IVFPQPrefilter,
    IVFPQPrefilterRoaring,
)

M_FACTOR = 1000  # Over-fetch factor for Post-filtering
K_FETCH_PREFILTER = 1000  # For Top-1 Accuracy
K_FETCH_POSTFILTER = 1 * M_FACTOR  # Over-fetch k for Post-filtering

SELECTIVITY_TARGETS = [
    ("Low (<1%)", 0.1, (0.0, 1.0)),
    ("Low-Mid (1-10%)", 1.0, (1.0, 10.0)),
    ("Mid (10-50%)", 10.0, (10.0, 50.0)),
    ("High (>50%)", 50.0, (50.0, 101.0)),  # Use 101 to be inclusive
]


def save_query_embeddings(
    path: str, queries: list[tuple[str, str, str, str]], vectors: np.ndarray
):
    """
    Saves queries, their filters, and their vectors to a Parquet file.
    """
    print(f"Saving query embeddings to {path}...")
    query_ids = [q[0] for q in queries]
    product_ids = [q[2] for q in queries]
    # Use "{}" as default for null/empty filters
    filters_list = [q[3] if q[3] else "{}" for q in queries]

    dim = vectors.shape[1]
    arr_qids = pa.array(query_ids)
    ar_asins = pa.array(product_ids)
    arr_filters = pa.array(filters_list)

    arr_values = pa.array(vectors.flatten())
    arr_list = pa.FixedSizeListArray.from_arrays(arr_values, list_size=dim)

    # 4. Build the table and write
    table = pa.Table.from_arrays(
        [arr_qids, ar_asins, arr_filters, arr_list],
        names=["query_id", "ground_truth_product_ids", "filters", "vector"],
    )
    pq.write_table(table, path)
    print("Save complete.")


def load_query_embeddings(
    path: str,
) -> tuple[list[tuple[int, int, str]], np.ndarray]:
    """
    Loads query embeddings and filters from a Parquet file.
    Returns:
        - List of (query_id, product_id, filters_json) tuples
        - Numpy array of vectors
    """
    print(f"Loading cached query embeddings from {path}...")
    df = pd.read_parquet(path)

    # Handle potential nulls from parquet
    df["filters"] = df["filters"].fillna("{}")

    qid_pid_filter_list = list(
        zip(
            df["query_id"].values.astype("int64"),
            df["ground_truth_product_ids"].values.astype("int64"),
            df["filters"].values,
        )
    )

    vectors = np.stack(df["vector"].values).astype("float32")

    print(f"Loaded {len(qid_pid_filter_list)} queries, filters, and vectors.")
    return qid_pid_filter_list, vectors


def main():
    parser = argparse.ArgumentParser(description="Run search evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["esci", "amz_c4"],
        default="esci",
        help="Dataset to use for queries: 'esci' or 'amz_c4' (default: esci)",
    )
    args = parser.parse_args()

    DB_PATH = "amz.db"
    EMBEDDINGS_PATH = "embeddings.parquet"
    # Use different cache path for different datasets
    QUERY_EMBEDDINGS_PATH = f"query_embeddings_{args.dataset}.parquet"

    K_GOAL = 1
    M_FACTOR = 100  # large over-fetch factor for postfiltering
    K_FETCH = K_GOAL * M_FACTOR

    SELECTIVITY_TARGETS = [
        ("Low (<1%)", 0.1, (0.0, 1.0)),
        ("Low-Mid (1-10%)", 1.0, (1.0, 10.0)),
        ("Mid (10-50%)", 10.0, (10.0, 50.0)),
        ("High (>50%)", 50.0, (50.0, 101.0)),  # Use 101 to be inclusive
    ]
    db = DB(path=DB_PATH)
    encoder = Encoder()
    # Use different bitmap file for different datasets
    ROARING_PATH = f"bitmaps_{args.dataset}.pkl"
    roaring = RoaringIndex(ROARING_PATH)
    vector_store = VectorStore(EMBEDDINGS_PATH, db)
    print("Building index...")
    flat_l2_index = FlatL2(
        vector_store.dims, vector_store.vectors, vector_store.product_ids
    )
    print("Built index, now building IVFPQ index...")
    ivfpq_index = IVFPQ(vector_store.vectors, vector_store.product_ids)
    print("IVFPQ index build complete.")

    if os.path.exists(QUERY_EMBEDDINGS_PATH):
        qid_pid_filter_list, all_query_vectors = load_query_embeddings(
            QUERY_EMBEDDINGS_PATH
        )
    else:
        print(f"No cache found. Encoding queries for dataset: {args.dataset}...")
        # Load queries based on selected dataset
        if args.dataset == "esci":
            queries = db.load_esci_queries()
        else:
            queries = db.load_amz_c4_queries()
        assert queries, "Exiting because no queries are found."

        query_texts = [q[1] for q in queries]
        sorted_indices = sorted(
            range(len(query_texts)), key=lambda k: len(query_texts[k])
        )
        sorted_queries = [queries[i] for i in sorted_indices]
        sorted_texts = [query_texts[i] for i in sorted_indices]

        all_query_vectors_sorted = encoder.encode_queries_in_batches(sorted_texts)
        print("Query encoding complete.")

        save_query_embeddings(
            QUERY_EMBEDDINGS_PATH, sorted_queries, all_query_vectors_sorted
        )

        qid_pid_filter_list = [
            (q[0], q[2], q[3] if q[3] else "[]") for q in sorted_queries
        ]
        all_query_vectors = all_query_vectors_sorted

    search_instances: List[Search] = []
    # ============================================================
    # 1) Raw prefilter: SQL filter → Flat ANN
    # ============================================================
    # search_instances.append(FlatPrefilter(db, encoder, vector_store))

    # ============================================================
    # 2) IVFPQ prefilter: SQL filter → IVFPQ ANN
    # ============================================================
    # search_instances.append(IVFPQPrefilter(db, encoder, vector_store))

    # ============================================================
    # 3) Roaring bitmap prefilter: Roaring filter → Flat ANN
    # ============================================================
    # search_instances.append(FlatPrefilterRoaring(db, encoder, vector_store, roaring))

    # ============================================================
    # 4) IVFPQ + Roaring bitmap prefilter: Roaring filter → IVFPQ ANN
    # ============================================================
    search_instances.append(IVFPQPrefilterRoaring(db, encoder, vector_store, roaring))

    # ============================================================
    # 5) Raw postfilter: Flat ANN → SQL filter
    # ============================================================
    # search_instances.append(FlatPostfilter(db, encoder, vector_store, flat_l2_index))

    # ============================================================
    # 6) IVFPQ postfilter: IVFPQ ANN → SQL filter
    # ============================================================
    # search_instances.append(IVFPQPostfilter(db, encoder, vector_store, ivfpq_index))

    # ============================================================
    # 7) Roaring bitmap postfilter: Flat ANN → Roaring filter
    # ============================================================
    # search_instances.append(
    #     FlatPostfilterRoaring(db, encoder, vector_store, flat_l2_index, roaring)
    # )

    # ============================================================
    # 8) IVFPQ + Roaring bitmap postfilter: IVFPQ → ANN Roaring filter
    # ============================================================
    # search_instances.append(
    #     IVFPQPostfilterRoaring(db, encoder, vector_store, ivfpq_index, roaring)
    # )

    for s in search_instances:
        result = s.evaluate(
            qid_pid_filter_list, all_query_vectors, SELECTIVITY_TARGETS, K_FETCH
        )
        s.log_results_summary(result, M_FACTOR, K_FETCH, "results.txt")

    db.close()


if __name__ == "__main__":
    main()
