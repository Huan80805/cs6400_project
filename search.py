import faiss
import pandas as pd
import numpy as np
import time
import sys
from typing import List, Tuple, Dict, Set, Optional
from db import DB
from encoder import Encoder
from vector_store import VectorStore


class Search:
    def __init__(self, db: DB, encoder: Encoder, parquet_path: str):
        print("starting search init")
        self.db = db
        self.encoder = encoder
        self.vector_store = VectorStore(path=parquet_path)
        # All in-memory, should probably persist
        print("before index init")
        self.index: Optional[faiss.IndexIDMap] = None
        print("old init done")
        self.ivfpq_index: Optional[faiss.Index] = None
        print("finished search init")

    def build_index(self):
        start_time = time.time()

        # default index with sequential ID
        default_idx = faiss.IndexFlatL2(self.vector_store.dims)

        # supports SQL PK
        self.index = faiss.IndexIDMap(default_idx)

        # add vectors to index with custom id
        # so that faiss returns the SQL PK as search results
        self.index.add_with_ids(
            self.vector_store.vectors, self.vector_store.product_ids
        )

        end_time = time.time()

        print(f"Index build time: {end_time - start_time:.2f} seconds")

    def build_ivfpq_index(
        self,
        nlist: Optional[int] = None,
        m: Optional[int] = None,
        nbits: int = 8,
        train_size: int = 200_000,
        nprobe: int = 16,
    ) -> None:
        """
        Build an IVFPQ index over all product vectors, in addition
        to the existing flat IndexFlatL2+IndexIDMap index.

        This does NOT change self.index (FlatL2); it populates self.ivfpq_index
        so you can compare FlatL2 vs IVFPQ side by side.
        nlist - Number of coarse clusters (inverted lists): more lists = finer partitioning and potentially faster search at the cost of a heavier index and training.
        m - Number of PQ subquantizers: controls how many equal sub-vectors each embedding is split into, trading off code length (memory/compute) vs quantization accuracy.
        nbits - Bits per subquantizer: sets how many centroids each subspace can use (e.g., 8 bits → 256 centroids), with more bits giving higher accuracy but slightly more memory and training cost.
        train_size - Number of vectors used to train the IVF and PQ codebooks: larger samples yield better centroids but increase training time.
        nprobe - Number of inverted lists probed per query at search time: higher values improve recall by scanning more clusters but increase latency.
        """
        start_time = time.time()

        xb = self.vector_store.vectors          # shape: (N, d)
        ids = self.vector_store.product_ids     # shape: (N,)
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

        self.ivfpq_index = ivfpq

        end_time = time.time()
        print(f"Index build time (IVFPQ): {end_time - start_time:.2f} seconds")


    def postfilter_search(
        self,
        query_vector: np.ndarray,
        k: int,
        filter: Dict,
    ) -> list[int]:
        assert self.index is not None, "Please call build_index() before searching."

        distances, ids = self.index.search(query_vector, k)
        candidate_ids = ids[0].tolist()

        filtered_allowed_set = self.db.get_filtered_ids(candidate_ids, filter)
        results = []
        for pid in candidate_ids:
            if pid in filtered_allowed_set:
                results.append(pid)

        return results

    def postfilter_search_ivfpq(
        self,
        query_vector: np.ndarray,
        k: int,
        filter: Dict,
    ) -> list[int]:
        """
        Same semantics as postfilter_search, but uses the IVFPQ index
        built by build_ivfpq_index().
        """
        assert (
            self.ivfpq_index is not None
        ), "Please call build_ivfpq_index() before IVFPQ searching."

        distances, ids = self.ivfpq_index.search(query_vector, k)
        candidate_ids = ids[0].tolist()

        filtered_allowed_set = self.db.get_filtered_ids(candidate_ids, filter)
        results = []
        for pid in candidate_ids:
            if pid in filtered_allowed_set:
                results.append(pid)

        return results
    
    def prefilter_search(
        self, query_vector: np.ndarray, k: int, filter: Dict
    ) -> Optional[int]:
        """
        Returns the single top product_id that matches, or None.
        """

        # get all ids matching filter

        # get all vectors matching these ids

        # construct index from the vectors (on-the-fly for each query)

        # search
        return None
