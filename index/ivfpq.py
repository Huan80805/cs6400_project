from index.base import Index, SearchResult
import time
import faiss
import numpy as np
from typing import Optional, List


class IVFPQ(Index):
    def __init__(
        self,
        vectors: np.ndarray,
        product_ids: List[int],
        nlist: Optional[int] = None,
        m: Optional[int] = None,
        nbits: int = 8,
        train_size: int = 200_000,
        nprobe: int = 16,
    ):
        """
        Build an IVFPQ index over all product vectors, in addition
        to the existing flat IndexFlatL2+IndexIDMap index.

        This does NOT change self.index (FlatL2); it populates self.ivfpq_index
        so you can compare FlatL2 vs IVFPQ side by side.
        """
        start_time = time.time()

        xb = vectors  # shape: (N, d)
        ids = product_ids  # shape: (N,)
        n_vectors, dim = xb.shape

        # --- Heuristic defaults if not provided ---
        if nlist is None:
            # Rule of thumb: ~sqrt(N), capped at 4096
            nlist = min(4096, max(1, int(np.sqrt(n_vectors))))

        if m is None:
            # Start from 64 and shrink until it divides dim
            m_candidate = min(64, dim)
            while m_candidate > 1 and dim % m_candidate != 0:
                m_candidate //= 2
            m = max(1, m_candidate)

        max_train_points = min(n_vectors, train_size)

        if nlist > max_train_points:
            print(
                f"Reducing nlist from {nlist} to {max_train_points} because "
                f"we only have {max_train_points} training points."
            )
            nlist = max_train_points

        # --- Clamp nbits so we don't have more PQ centroids than training points ---
        # PQ uses k = 2**nbits clusters per sub-quantizer. FAISS requires nx >= k.
        max_clusters = max_train_points
        if max_clusters < (1 << nbits):
            old_nbits = nbits
            # largest nbits such that 2**nbits <= max_clusters
            nbits = int(np.floor(np.log2(max_clusters)))
            nbits = max(1, nbits)
            print(
                f"Reducing nbits from {old_nbits} to {nbits} because we only have "
                f"{max_clusters} training points."
            )

        print(
            f"Building IVFPQ index: N={n_vectors}, dim={dim}, "
            f"nlist={nlist}, m={m}, nbits={nbits}"
        )

        # --- Quantizer + IVFPQ index ---
        quantizer = faiss.IndexFlatL2(dim)
        ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        # --- Training data selection ---
        if n_vectors > train_size:
            idx = np.random.choice(n_vectors, train_size, replace=False)
            train_x = xb[idx]
            print(f"Training IVFPQ on a subsample of {train_size} vectors...")
        else:
            train_x = xb
            print(f"Training IVFPQ on all {n_vectors} vectors...")

        ivfpq.train(train_x)
        print("IVFPQ training complete. Adding vectors to IVFPQ index...")

        # --- Add all vectors with explicit IDs ---
        ivfpq.add_with_ids(xb, ids)

        # --- Search-time params ---
        ivfpq.nprobe = min(nprobe, nlist)

        self.index = ivfpq

        end_time = time.time()
        print(f"Index build time (IVFPQ): {end_time - start_time:.2f} seconds")

    def search(self, query_vector: np.ndarray, k: int, *args, **kwargs) -> SearchResult:
        _, ids = self.index.search(query_vector, k, *args, **kwargs)
        candidate_ids = [int(i) for i in ids[0].tolist() if int(i) != -1]
        return candidate_ids
