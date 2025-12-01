from typing import Optional
from db import DB
import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from encoder import Encoder


class QueryVectorStore:
    def __init__(
        self,
        query_dataset_name: str,
        encoder: Encoder,
        db: Optional[DB] = None,
    ):
        self.query_dataset_name = query_dataset_name
        self.embedding_file_path = f"query_embeddings_{query_dataset_name}.parquet"
        self.encoder = encoder
        self.db = db

    def read_query_embeddings_from_cache(
        self,
    ) -> tuple[list[tuple[int, int, str]], np.ndarray]:
        """
        Loads query embeddings and filters from a Parquet file.
        Returns:
            - List of (query_id, product_id, filters_json) tuples
            - Numpy array of vectors
        """
        print(f"Loading cached query embeddings from {self.embedding_file_path}...")
        df = pd.read_parquet(self.embedding_file_path)

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

    def save_query_embeddings(
        self, queries: list[tuple[str, str, str, str]], vectors: np.ndarray
    ):
        """
        Saves queries, their filters, and their vectors to a Parquet file.
        """
        print(f"Saving query embeddings to {self.embedding_file_path}...")
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
        pq.write_table(table, self.embedding_file_path)
        print("Save complete.")

    def load_query_embeddings(self) -> tuple[list[tuple[int, int, str]], np.ndarray]:
        if os.path.exists(self.embedding_file_path):
            qid_pid_filter_list, all_query_vectors = (
                self.read_query_embeddings_from_cache(self.embedding_file_path)
            )
        else:
            print(
                f"No cache found. Encoding queries for dataset: {self.query_dataset_name}..."
            )
            # Load queries based on selected dataset
            if self.query_dataset_name == "esci":
                queries = self.db.load_esci_queries()
            else:
                queries = self.db.load_amz_c4_queries()
            assert queries, "Exiting because no queries are found."

            query_texts = [q[1] for q in queries]
            sorted_indices = sorted(
                range(len(query_texts)), key=lambda k: len(query_texts[k])
            )
            sorted_queries = [queries[i] for i in sorted_indices]
            sorted_texts = [query_texts[i] for i in sorted_indices]

            all_query_vectors_sorted = self.encoder.encode_queries_in_batches(
                sorted_texts
            )
            print("Query encoding complete.")

            self.save_query_embeddings(sorted_queries, all_query_vectors_sorted)

            qid_pid_filter_list = [
                (q[0], q[2], q[3] if q[3] else "[]") for q in sorted_queries
            ]
            all_query_vectors = all_query_vectors_sorted
        return qid_pid_filter_list, all_query_vectors
