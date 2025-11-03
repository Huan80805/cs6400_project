import numpy as np
import time
import sys
import os
from tqdm import tqdm

# New imports for optimized Parquet I/O
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


def main():
    DB_PATH = "amz.db"
    EMBEDDINGS_PATH = "embeddings.parquet"
    QUERY_EMBEDDINGS_PATH = "query_embeddings.parquet"
    QUERY_BATCH_SIZE = 256  # Use a larger batch size

    K_GOAL = 1
    M_FACTOR = 1000
    K_FETCH = K_GOAL * M_FACTOR

    # This static filter is no longer used, we use a dynamic one per-query

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
        queries = db.load_esci_queries()
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

        # Create the 3-tuple list: (qid, pid, filters_json)
        qid_pid_filter_list = [
            (q[0], q[2], q[3] if q[3] else "{}") for q in sorted_queries
        ]
        all_query_vectors = all_query_vectors_sorted

    latencies_ms = []
    hits = 0

    gtsims = []
    for i in tqdm(range(len(qid_pid_filter_list)), desc="Evaluating queries"):
        _, ground_truth_product_id_str, filters_json = qid_pid_filter_list[i]
        ground_truth_product_id = int(ground_truth_product_id_str)  # Ensure it's int

        query_vector = all_query_vectors[i : i + 1, :]  # Keep 2D shape

        # --- DYNAMIC FILTER PREPARATION ---
        # 1. Parse the JSON string
        try:
            loaded_filter_dict = json.loads(filters_json)
        except json.JSONDecodeError:
            loaded_filter_dict = {}  # Default to empty if JSON is invalid

        # 2. Convert inner lists (from JSON) to tuples (for db.py)
        # Assumes filters_json is like: '{"average_rating": [">=", 4.0]}'
        # Converts to: {"average_rating": (">=", 4.0)}
        dynamic_filter = {
            key: tuple(op_val)
            for key, op_val in loaded_filter_dict.items()
            if isinstance(op_val, list)
        }

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

        gt_sim = (query_vector @ gt_vector.T)[0, 0] if gt_vector is not None else -1.0
        gtsims.append(gt_sim)

        is_hit = ground_truth_product_id in final_result_set

        # Recall@1 logic: Check if the top result (if any) is the GT
        if is_hit:
            hits += 1

    total_queries = len(qid_pid_filter_list)
    recall_at_1 = (hits / total_queries) if total_queries > 0 else 0
    avg_latency = np.mean(latencies_ms) if latencies_ms else 0
    p95_latency = np.percentile(latencies_ms, 95) if latencies_ms else 0

    print(f"Total Queries:     {total_queries}")
    print(f"Total Hits:        {hits}")
    print(f"Recall@1:          {recall_at_1:.4f}")
    print(f"Avg. Latency (ms): {avg_latency:.2f}")
    print(f"P95 Latency (ms):  {p95_latency:.2f}")
    avg_gt_sim = np.mean(gtsims) if gtsims else 0
    median_gt_sim = np.median(gtsims) if gtsims else 0
    print(f"Avg. GT Sim:       {avg_gt_sim:.4f}")

    # --- 5. Clean up ---
    db.close()


if __name__ == "__main__":
    main()
