from .base import Search
from index import FlatL2
import numpy as np


class UnfilteredFlat(Search):
    def __init__(
        self,
        db,
        encoder,
        vector_store,
        method_title="Unfiltered (Flat)",
        method_name="Unfiltered (Flat)",
    ):
        super().__init__(db, encoder, vector_store, method_title, method_name)

    def build_index(self):
        self.index = FlatL2(
            self.vector_store.dims,
            self.vector_store.vectors,
            self.vector_store.product_ids,
        )

    def search(self, query_vector: np.ndarray, k: int):
        """
        Baseline: no filter, exact ANN over the full flat index.
        """
        assert self.index is not None, (
            "Please call self.build_index() before searching."
        )
        return self.index.search(query_vector, k)
