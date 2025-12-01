from .base import Search
import numpy as np
from typing import List, Dict, Any


class FlatPostfilter(Search):
    def __init__(
        self,
        db,
        encoder,
        vector_store,
        index,
        method_title="POST-FILTERING BY SELECTIVITY",
        method_name="Post-Filter (Flat)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name, index)

    @property
    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        assert self.index is not None, "self.index is not present"

        candidate_ids = self.index.search(query_vector, k)

        filtered_allowed_set = self.db.get_filtered_ids(candidate_ids, filter)
        return [pid for pid in candidate_ids if pid in filtered_allowed_set]
