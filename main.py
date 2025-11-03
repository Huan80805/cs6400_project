import numpy as np
import time
import sys
import os
from tqdm import tqdm

# New imports for optimized Parquet I/O
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Optional

# Import our classes
from db import DB
from encoder import Encoder
from search import Search
import json


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


def select_filter_by_selectivity(
    filter_list: list[dict[str, Any]],
    target_percent: float,
    selectivity_range: tuple[float, float],
) -> Optional[dict[str, Any]]:
    """
    Selects one filter from the list that falls within the valid
    selectivity_range AND is closest to the target_percent.
    """
    if not filter_list:
        return None

    min_perc, max_perc = selectivity_range

    # 1. Find all valid filters *within the correct bucket*
    valid_filters = []
    for f in filter_list:
        perc = f.get("match_percentage")
        if perc is not None and min_perc <= perc < max_perc:
            valid_filters.append(f)

    # 2. If no filters are in this bucket, we can't use one.
    if not valid_filters:
        return None

    # 3. From the valid subset, find the one closest to our ideal target.
    selected_filter = min(
        valid_filters,
        key=lambda f: abs(f.get("match_percentage", 101) - target_percent),
    )
    return selected_filter


def build_filter_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Takes: {"filter_column": "average_rating", "filter_value": [3.5, 3.7], ...}
    Returns: {"average_rating": ("BETWEEN", (3.5, 3.7))}
    """
    if not spec:
        return {}

    col = spec.get("filter_column")
    val = spec.get("filter_value")

    if col is None or val is None:
        return {}

    # Handle range/list values as BETWEEN
    if isinstance(val, list) and len(val) == 2:
        return {col: ("BETWEEN", tuple(val))}

    # Handle JSON/text LIKE searches
    if col in ("features_json", "details_json"):
        return {col: ("LIKE", f"%{val}%")}

    # Default to simple equality
    return {col: ("=", val)}


def main():
    DB_PATH = "amz.db"
    EMBEDDINGS_PATH = "embeddings.parquet"
    QUERY_EMBEDDINGS_PATH = "query_embeddings.parquet"
    QUERY_BATCH_SIZE = 256

    K_GOAL = 1
    M_FACTOR = 1000  # large over-fetch factor for postfiltering
    K_FETCH = K_GOAL * M_FACTOR

    SELECTIVITY_TARGETS = [
        ("Low (<1%)", 0.1, (0.0, 1.0)),
        ("Low-Mid (1-10%)", 1.0, (1.0, 10.0)),
        ("Mid (10-50%)", 10.0, (10.0, 50.0)),
        ("High (>50%)", 50.0, (50.0, 101.0)),  # Use 101 to be inclusive
    ]

    db = DB(path=DB_PATH)
    encoder = Encoder()
    search = Search(db=db, encoder=encoder, parquet_path=EMBEDDINGS_PATH)

    search.build_index()

    if os.path.exists(QUERY_EMBEDDINGS_PATH):
        qid_pid_filter_list, all_query_vectors = load_query_embeddings(
            QUERY_EMBEDDINGS_PATH
        )
    else:
        print("No cache found. Encoding queries...")
        queries = db.load_esci_queries()  # This now returns 4-tuples
        assert queries, "Exiting because no queries are found."

        query_texts = [q[1] for q in queries]
        sorted_indices = sorted(
            range(len(query_texts)), key=lambda k: len(query_texts[k])
        )
        sorted_queries = [queries[i] for i in sorted_indices]
        sorted_texts = [query_texts[i] for i in sorted_indices]

        all_query_vectors_sorted = encoder.encode_queries_in_batches(
            sorted_texts, batch_size=QUERY_BATCH_SIZE
        )
        print("Query encoding complete.")

        save_query_embeddings(
            QUERY_EMBEDDINGS_PATH, sorted_queries, all_query_vectors_sorted
        )

        # Create the 3-tuple list: (qid, pid, filters_json_string)
        qid_pid_filter_list = [
            (q[0], q[2], q[3] if q[3] else "[]") for q in sorted_queries
        ]
        all_query_vectors = all_query_vectors_sorted

    all_results = []  # Store results for each level

    for level_name, target_percent, selectivity_range in SELECTIVITY_TARGETS:
        latencies_ms = []
        hits = 0
        gtsims = []
        total_queries = 0

        for i in tqdm(
            range(len(qid_pid_filter_list)),
            desc=f"Evaluating Post-Filter (Selectivity ~{target_percent}%)",
        ):
            query_id, ground_truth_product_id_str, filters_json_string = (
                qid_pid_filter_list[i]
            )
            ground_truth_product_id = int(ground_truth_product_id_str)

            query_vector = all_query_vectors[i : i + 1, :]

            try:
                filter_suite = json.loads(filters_json_string)
            except json.JSONDecodeError:
                filter_suite = []  # Default to empty if JSON is invalid

            # Select filter based on the *current* loop's target_percent
            selected_spec = select_filter_by_selectivity(
                filter_suite, target_percent, selectivity_range
            )
            dynamic_filter = build_filter_from_spec(selected_spec)

            if not dynamic_filter:
                continue

            total_queries += 1
            start_time = time.perf_counter()

            final_result_pids = search.postfilter_search(
                query_vector=query_vector,
                k=K_FETCH,
                filter=dynamic_filter,
            )

            final_result_set = set(final_result_pids)
            end_time = time.perf_counter()
            latencies_ms.append((end_time - start_time) * 1000)

            gt_vector = search.vector_store.get_vector_by_product_id(
                ground_truth_product_id
            )

            gt_sim = (
                (query_vector @ gt_vector.T)[0, 0] if gt_vector is not None else -1.0
            )
            gtsims.append(gt_sim)

            is_hit = ground_truth_product_id in final_result_set

            if is_hit:
                hits += 1

        recall = (hits / total_queries) if total_queries > 0 else 0
        avg_latency = np.mean(latencies_ms) if latencies_ms else 0
        p95_latency = np.percentile(latencies_ms, 95) if latencies_ms else 0
        avg_gt_sim = np.mean(gtsims) if gtsims else 0

        all_results.append(
            {
                "level": level_name,
                "target_selectivity": f"~{target_percent}%",
                "total_queries": total_queries,
                "hits": hits,
                "recall": recall,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "average_cosine_similarity_between_query_vector_and_ground_truth_item": avg_gt_sim,
            }
        )

    print("\n--- FINAL SUMMARY: POST-FILTERING BY SELECTIVITY ---")
    print("-" * 70)
    print(f"M_FACTOR (Overfetch): {M_FACTOR} (K_FETCH={K_FETCH})")
    print(
        f"{'Level':<18} | {'Recall':<8} | {'P95 Lat (ms)':<12} | {'Avg Lat (ms)':<12} | {'Hits':<5}"
    )
    print("-" * 70)

    for metrics in all_results:
        print(
            f"{metrics['level']:<18} | {metrics['recall']:<8.4f} | {metrics['p95_latency_ms']:<12.2f} | {metrics['avg_latency_ms']:<12.2f} | {metrics['hits']:<5}"
        )

    print("-" * 70)

    db.close()


if __name__ == "__main__":
    main()
