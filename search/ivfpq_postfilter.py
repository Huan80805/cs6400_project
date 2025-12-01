from .base import Search
from index import IVFPQ
import numpy as np
from typing import Dict


class IVFPQPostfilter(Search):
    def __init__(
        self,
        db,
        encoder,
        vector_store,
        index: IVFPQ,
        method_title="POST-FILTERING BY SELECTIVITY (IVFPQ)",
        method_name="Post-Filter (IVFPQ)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name, index)

    def search(self, query_vector: np.ndarray, k: int, filter: Dict):
        assert self.index is not None, "self.index is not present"

        candidate_ids = self.index.search(query_vector, k)

        filtered_allowed_set = self.db.get_filtered_ids(candidate_ids, filter)

        return [pid for pid in candidate_ids if pid in filtered_allowed_set]
