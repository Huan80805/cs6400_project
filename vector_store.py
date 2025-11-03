import pandas as pd
import numpy as np


class VectorStore:
    def __init__(
        self,
        path: str,
        filters: dict | None = None,
    ):
        if not filters:
            self.df = pd.read_parquet(path)
        elif filters.get("product_id"):
            self.df = pd.read_parquet(
                path, filters=[("product_id", "in", filters["product_id"])]
            )

        self.vectors = np.stack(self.df["vector"].values).astype("float32")
        self.product_ids = self.df["product_id"].values.astype("int64")
        self.dims = int(self.vectors.shape[1])
        self.product_id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

    def get_vector_by_product_id(self, product_id: int) -> np.ndarray | None:
        if product_id not in self.product_id_to_idx:
            return None
        vector_index = self.product_id_to_idx[product_id]
        return self.vectors[vector_index : vector_index + 1, :]
