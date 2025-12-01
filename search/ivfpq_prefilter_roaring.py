from .base import Search
from index import IVFPQ
import numpy as np
from typing import Dict, List
from bitmap import RoaringIndex
from db import DB
from encoder import Encoder
from vector_store import VectorStore
import faiss


class IVFPQPrefilterRoaring(Search):
    def __init__(
        self,
        db: DB,
        encoder: Encoder,
        vector_store: VectorStore,
        roaring_index: RoaringIndex,
        index: IVFPQ = None,
        method_title="PRE-FILTERING BY SELECTIVITY (IVFPQ + Roaring)",
        method_name="Pre-Filter (IVFPQ + Roaring)",
        rebuild_index: bool = True,
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name, index)
        self.rebuild_index = rebuild_index
        self.roaring_index = roaring_index

    def build_index(self, vectors: np.ndarray, product_ids: List[int]):
        return IVFPQ(
            vectors=vectors,
            product_ids=product_ids,
        )

    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        """
        1. Roaring Filter -> IDs
        2. Build Temp IVFPQ Index on Subset using reusable build_ivfpq_index
        3. Search
        """
        assert self.roaring_index is not None, "RoaringIndex not configured."

        # 1. Get ALL ids matching the filter from Roaring Bitmaps
        filtered_ids_set = self.roaring_index.get_ids_for_filter(filter)
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

            # 3. Construct temporary IVFPQ index
            temp_index: IVFPQ = self.build_index(
                vectors=sub_vectors, product_ids=sub_ids
            )

            # 4. Search
            search_k = min(k, len(sub_ids))
            return temp_index.search(query_vector, search_k)
        else:
            assert self.index is not None, "self.index is not present"
            sel = faiss.IDSelectorBatch(list(filtered_ids_set))
            params = faiss.SearchParametersIVF(sel=sel)
            return self.index.search(query_vector, k, params=params)
