from .base import Search
from index import IVFPQ
import numpy as np
from typing import Dict, List


class IVFPQPrefilter(Search):
    def __init__(
        self,
        db,
        encoder,
        vector_store,
        method_title="PRE-FILTERING BY SELECTIVITY (IVFPQ)",
        method_name="Pre-Filter (IVFPQ)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name)

    @property
    def build_index(self, vectors: np.ndarray, product_ids: List[int]):
        return IVFPQ(
            vectors=vectors,
            product_ids=product_ids,
        )

    @property
    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        """
        1. SQL Filter -> IDs
        2. Build Temp IVFPQ Index on Subset using reusable build_ivfpq_index
        3. Search
        """
        # 1. Get ALL ids matching the filter from the DB
        filtered_ids_set = self.db.get_all_ids_matching_filter(filter)

        # 2. Get vectors
        sub_vectors, sub_ids = (
            self.vector_store.get_vectors_and_product_ids_from_prefiltered_ids(
                filtered_ids_set
            )
        )
        n_sub, _ = sub_vectors.shape
        if n_sub == 0:
            return []

        # 3. Construct temporary IVFPQ index REUSING the generalized method
        #    We allow the method to calculate nlist/m heuristics automatically
        temp_index: IVFPQ = self.build_index(sub_vectors, sub_ids)

        # 4. Search
        search_k = min(k, n_sub)
        return temp_index.search(query_vector, search_k)
