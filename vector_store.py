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
