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


def save_query_embeddings(
    path: str, queries: list[tuple[str, str, str]], vectors: np.ndarray
):
    """
    Saves queries and their vectors to a Parquet file using the
    optimized pyarrow FixedSizeListArray method.
    """
    print(f"Saving query embeddings to {path}...")
    query_ids = [q[0] for q in queries]
    product_ids = [q[2] for q in queries]
    dim = vectors.shape[1]
    arr_qids = pa.array(query_ids)
    ar_asins = pa.array(product_ids)

    arr_values = pa.array(vectors.flatten())
    arr_list = pa.FixedSizeListArray.from_arrays(arr_values, list_size=dim)

    # 4. Build the table and write
    table = pa.Table.from_arrays(
        [arr_qids, ar_asins, arr_list],
        names=["query_id", "ground_truth_product_ids", "vector"],
    )
    pq.write_table(table, path)
    print("Save complete.")


def load_query_embeddings(
    path: str,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Loads query embeddings from a Parquet file.
    Reading with pandas is simple and efficient.
    """
    print(f"Loading cached query embeddings from {path}...")
    df = pd.read_parquet(path)
    qid_pid_pair_list = list(
        zip(
            df["query_id"].values.astype("int64"),
            df["ground_truth_product_ids"].values.astype("int64"),
        )
    )

    vectors = np.stack(df["vector"].values).astype("float32")

    print(f"Loaded {len(qid_pid_pair_list)} queries and vectors.")
    return qid_pid_pair_list, vectors


def main():
    DB_PATH = "amz.db"
    EMBEDDINGS_PATH = "embeddings.parquet"
    QUERY_EMBEDDINGS_PATH = "query_embeddings.parquet"
    QUERY_BATCH_SIZE = 256  # Use a larger batch size

    K_GOAL = 1
    M_FACTOR = 10
    K_FETCH = K_GOAL * M_FACTOR

    FILTER = {"average_rating": (">=", 4.0)}  # Example filter

    db = DB(path=DB_PATH)
    encoder = Encoder()
    search = Search(db=db, encoder=encoder, parquet_path=EMBEDDINGS_PATH)

    search.build_index()

    if os.path.exists(QUERY_EMBEDDINGS_PATH):
        qid_pid_pair_list, all_query_vectors = load_query_embeddings(
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
        qid_pid_pair_list = [(q[0], q[2]) for q in sorted_queries]
        all_query_vectors = all_query_vectors_sorted

    latencies_ms = []
    hits = 0

    for i in tqdm(range(len(qid_pid_pair_list)), desc="Evaluating queries"):
        query_id, ground_truth_product_id = qid_pid_pair_list[i]
        query_vector = all_query_vectors[i : i + 1, :]  # Keep 2D shape
        start_time = time.perf_counter()
        final_result_pid = search.postfilter_search(
            query_vector=query_vector,
            k=K_FETCH,
            filter=FILTER,
        )
        final_result_pid = set(final_result_pid)
        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000)

        print("Result:")
        print(
            f"ground_truth_product_id: {ground_truth_product_id}, matched: {int(ground_truth_product_id) in final_result_pid},  query_id: {query_id}"
        )

        # This has not been updated,
        # refer to the print result above
        # Recall@1
        # if final_result_pid is not None:
        #     if str(final_result_pid) == str(ground_truth_product_id):
        #         hits += 1

    total_queries = len(qid_pid_pair_list)
    recall_at_1 = (hits / total_queries) if total_queries > 0 else 0
    avg_latency = np.mean(latencies_ms) if latencies_ms else 0
    p95_latency = np.percentile(latencies_ms, 95) if latencies_ms else 0

    print(f"Total Queries:     {total_queries}")
    print(f"Total Hits:        {hits}")
    print(f"Recall@1:          {recall_at_1:.4f}")
    print(f"Avg. Latency (ms): {avg_latency:.2f}")
    print(f"P95 Latency (ms):  {p95_latency:.2f}")

    # --- 5. Clean up ---
    db.close()


if __name__ == "__main__":
    main()
