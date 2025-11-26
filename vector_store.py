import pandas as pd
import numpy as np
from typing import Optional
from db import DB

class VectorStore:
    def __init__(
        self,
        path: str,
        db: Optional[DB] = None,
    ):
        self.df = pd.read_parquet(path)
        
        # If DB is provided, map parent_asin to current product_id
        if db and "parent_asin" in self.df.columns:
            print("Mapping embeddings using parent_asin -> product_id from DB...")
            
            # 1. Get current mapping from DB
            cur = db.conn.cursor()
            cur.execute("SELECT parent_asin, product_id FROM products")
            asin_to_pid = {row[0]: row[1] for row in cur.fetchall()}
            
            # 2. Filter and map the dataframe
            # Create a new column 'mapped_pid'
            self.df["mapped_pid"] = self.df["parent_asin"].map(asin_to_pid)
            
            # Drop rows where ASIN is not in the current DB
            initial_len = len(self.df)
            self.df = self.df.dropna(subset=["mapped_pid"])
            dropped_count = initial_len - len(self.df)
            if dropped_count > 0:
                print(f"Dropped {dropped_count} embeddings not found in current DB.")
            
            # Use the mapped PID as the product_id
            self.product_ids = self.df["mapped_pid"].values.astype("int64")
        else:
            # Fallback to existing product_id if no DB or no parent_asin column
            print("Using existing product_id from parquet (no mapping).")
            self.product_ids = self.df["product_id"].values.astype("int64")

        self.vectors = np.stack(self.df["vector"].values).astype("float32")
        self.dims = int(self.vectors.shape[1])
        self.product_id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

    def get_vector_by_product_id(self, product_id: int) -> np.ndarray | None:
        if product_id not in self.product_id_to_idx:
            return None
        vector_index = self.product_id_to_idx[product_id]
        return self.vectors[vector_index : vector_index + 1, :]
