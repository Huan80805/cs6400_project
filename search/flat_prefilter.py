from .base import Search
from index import FlatL2
import numpy as np
from typing import List, Dict, Any


class FlatPrefilter(Search):
    def __init__(
        self,
        db,
        encoder,
        vector_store,
        method_title="PRE-FILTERING BY SELECTIVITY",
        method_name="Pre-Filter (Flat)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name)

    @property
    def build_index(self, vectors: np.ndarray, product_ids: List[int]):
        # don't assign to self.index since it is temporary, on-the-fly
        return FlatL2(self.vector_store.dims, vectors, product_ids)

    @property
    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        filtered_ids_set = self.db.get_all_ids_matching_filter(filter)
        if not filtered_ids_set:
            return []

        sub_vectors, sub_ids = (
            self.vector_store.get_vectors_and_product_ids_from_prefiltered_ids(
                filtered_ids_set
            )
        )
        if len(sub_ids) == 0:
            return []

        temp_index: FlatL2 = self.build_index(sub_vectors, sub_ids)
        search_k = min(k, len(sub_ids))
        if search_k == 0:
            return []

        return temp_index.search(query_vector, search_k)
