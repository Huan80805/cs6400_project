from .base import Search
from index import FlatL2
import numpy as np
from typing import Dict
from bitmap import RoaringIndex
from db import DB
from encoder import Encoder
from vector_store import VectorStore
from index import FlatL2


class FlatPostfilterRoaring(Search):
    def __init__(
        self,
        db: DB,
        encoder: Encoder,
        vector_store: VectorStore,
        index: FlatL2,
        roaring_index: RoaringIndex,
        method_title="POST-FILTERING BY SELECTIVITY (Roaring)",
        method_name="Post-Filter (Roaring)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name, index)
        self.roaring_index = roaring_index

    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        """
        Post-filtering using Roaring bitmaps instead of SQL:
        1) vector search over the full flat index,
        2) intersect the candidate IDs with a Roaring bitmap for the filter.
        """
        assert self.index is not None, "self.index is not present"
        assert self.roaring_index is not None, "RoaringIndex not configured."

        candidate_ids = self.index.search(query_vector, k)

        # Roaring: precomputed set of **allowed** product_ids for this filter
        allowed_ids = self.roaring_index.get_ids_for_filter(filter)  # Set[int]

        return [pid for pid in candidate_ids if pid in allowed_ids]
