from index.base import Index, SearchResult
import time
import faiss
import numpy as np
from typing import List, Set


class FlatL2(Index):
    def __init__(self, dims: int, vectors: np.ndarray, product_ids: List[int]):
        start_time = time.time()

        # default index with sequential ID
        default_idx = faiss.IndexFlatL2(dims)

        # supports SQL PK
        self.index = faiss.IndexIDMap(default_idx)

        # add vectors to index with custom id
        # so that faiss returns the SQL PK as search results
        self.index.add_with_ids(vectors, product_ids)

        end_time = time.time()
        print(f"Index build time: {end_time - start_time:.2f} seconds")

    def search(self, query_vector: np.ndarray, k: int) -> SearchResult:
        _, ids = self.index.search(query_vector, k)
        candidate_ids = [int(i) for i in ids[0].tolist() if int(i) != -1]
        return candidate_ids
