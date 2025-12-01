from .base import Search
from index import IVFPQ
import numpy as np
from typing import Dict
from bitmap import RoaringIndex
from db import DB
from encoder import Encoder
from vector_store import VectorStore


class IVFPQPostfilterRoaring(Search):
    def __init__(
        self,
        db: DB,
        encoder: Encoder,
        vector_store: VectorStore,
        index: IVFPQ,
        roaring_index: RoaringIndex,
        method_title="POST-FILTERING BY SELECTIVITY (IVFPQ + Roaring)",
        method_name="Post-Filter (IVFPQ + Roaring)",
    ):
        super().__init__(db, encoder, vector_store, index, method_title, method_name)
        self.roaring_index = roaring_index

    @property
    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        """
        1) Global IVFPQ ANN over all vectors
        2) Post-filter the candidate IDs using Roaring bitmaps
        """
        assert self.index is not None, "self.index is not present"
        assert self.roaring_index is not None, "RoaringIndex not configured."

        candidate_ids = self.index.search(query_vector, k)

        allowed_ids = self.roaring_index.get_ids_for_filter(filter)
        return [pid for pid in candidate_ids if pid in allowed_ids]
