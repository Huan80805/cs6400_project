import faiss
import pandas as pd
import numpy as np
import time
import sys
from typing import List, Tuple, Dict, Set, Optional
from db import DB
from encoder import Encoder
from vector_store import VectorStore


class Search:
    def __init__(self, db: DB, encoder: Encoder, parquet_path: str):
        self.db = db
        self.encoder = encoder
        self.vector_store = VectorStore(path=parquet_path)
        # All in-memory, should probably persist
        self.index: Optional[faiss.IndexIDMap] = None

    def build_index(self):
        start_time = time.time()

        # default index with sequential ID
        default_idx = faiss.IndexFlatL2(self.vector_store.dims)

        # supports SQL PK
        self.index = faiss.IndexIDMap(default_idx)

        # add vectors to index with custom id
        # so that faiss returns the SQL PK as search results
        self.index.add_with_ids(
            self.vector_store.vectors, self.vector_store.product_ids
        )

        end_time = time.time()

        print(f"Index build time: {end_time - start_time:.2f} seconds")

    def postfilter_search(
        self,
        query_vector: np.ndarray,
        k: int,
        filter: Dict,
    ) -> list[int]:
        assert self.index is not None, "Please call build_index() before searching."

        distances, ids = self.index.search(query_vector, k)
        candidate_ids = ids[0].tolist()

        filtered_allowed_set = self.db.get_filtered_ids(candidate_ids, filter)
        results = []
        for pid in candidate_ids:
            if pid in filtered_allowed_set:
                results.append(pid)

        return results

    def prefilter_search(
        self, query_vector: np.ndarray, k: int, filter: Dict
    ) -> Optional[int]:
        """
        Returns the single top product_id that matches, or None.
        """

        # get all ids matching filter

        # get all vectors matching these ids

        # construct index from the vectors (on-the-fly for each query)

        # search
        return None
