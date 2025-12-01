from .base import Search
from index import IVFPQ
import numpy as np
from typing import Dict, List
import faiss


class IVFPQPrefilter(Search):
    def __init__(
        self,
        db,
        encoder,
        vector_store,
        index: IVFPQ = None,
        method_title="PRE-FILTERING BY SELECTIVITY (IVFPQ)",
        method_name="Pre-Filter (IVFPQ)",
        rebuild_index: bool = True,
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name, index)
        self.rebuild_index = rebuild_index

    def build_index(self, vectors: np.ndarray, product_ids: List[int]):
        return IVFPQ(
            vectors=vectors,
            product_ids=product_ids,
        )

    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        """
        1. SQL Filter -> IDs
        2. Build Temp IVFPQ Index on Subset using reusable build_ivfpq_index
        3. Search
        """
        # 1. Get ALL ids matching the filter from the DB
        filtered_ids_set = self.db.get_all_ids_matching_filter(filter)
        if not filtered_ids_set:
            return []

        if self.rebuild_index:
            # 2. Get vectors
            sub_vectors, sub_ids = (
                self.vector_store.get_vectors_and_product_ids_from_prefiltered_ids(
                    filtered_ids_set
                )
            )
            if len(sub_ids) == 0:
                return []

            # 3. Construct temporary IVFPQ index REUSING the generalized method
            #    We allow the method to calculate nlist/m heuristics automatically
            temp_index: IVFPQ = self.build_index(sub_vectors, sub_ids)

            # 4. Search
            search_k = min(k, len(sub_ids))
            return temp_index.search(query_vector, search_k)
        else:
            assert self.index is not None, "self.index is not present"
            sel = faiss.IDSelectorBatch(list(filtered_ids_set))
            params = faiss.SearchParametersIVF(sel=sel)
            return self.index.search(query_vector, k, params=params)
