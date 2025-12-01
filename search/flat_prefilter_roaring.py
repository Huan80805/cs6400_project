from .base import Search
from index import FlatL2
import numpy as np
from typing import Dict, List
from bitmap import RoaringIndex
from db import DB
from encoder import Encoder
from vector_store import VectorStore


class FlatPrefilterRoaring(Search):
    def __init__(
        self,
        db: DB,
        encoder: Encoder,
        vector_store: VectorStore,
        roaring_index: RoaringIndex,
        method_title="PRE-FILTERING BY SELECTIVITY (Roaring)",
        method_name="Pre-Filter (Roaring)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name)
        self.roaring_index = roaring_index

    @property
    def build_index(self, vectors: np.ndarray, product_ids: List[int]):
        return FlatL2(
            vectors=vectors,
            product_ids=product_ids,
        )

    @property
    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        """
        1. Roaring Filter -> IDs
        2. Build Temp Flat Index on Subset using reusable build_index
        3. Search
        """
        assert self.roaring_index is not None, "RoaringIndex not configured."

        # 1. Get ALL ids matching the filter from Roaring Bitmaps
        filtered_ids_set = self.roaring_index.get_ids_for_filter(filter)

        # 2. Get vectors
        sub_vectors, sub_ids = (
            self.vector_store.get_vectors_and_product_ids_from_prefiltered_ids(
                filtered_ids_set
            )
        )
        if len(sub_ids) == 0:
            return []

        # 3. Construct a temporary brute-force index
        temp_index: FlatL2 = self.build_index(vectors=sub_vectors, ids=sub_ids)

        # 4. Search
        search_k = min(k, len(sub_ids))
        return temp_index.search(query_vector, search_k)
