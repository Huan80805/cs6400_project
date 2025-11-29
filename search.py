import faiss
import pandas as pd
import numpy as np
import time
import sys
from typing import List, Tuple, Dict, Set, Optional
from db import DB
from encoder import Encoder
from vector_store import VectorStore
from roaring_index import RoaringIndex

# IMPORTANT: every search method must return a ranked list!!!
class Search:
    def __init__(
        self,
        db: DB,
        encoder: Encoder,
        parquet_path: str,
        roaring_index: Optional[RoaringIndex] = None,
    ):
        self.db = db
        self.encoder = encoder
        self.vector_store = VectorStore(path=parquet_path, db=db)
        # All in-memory, should probably persist
        self.index: Optional[faiss.IndexIDMap] = None
        self.ivfpq_index: Optional[faiss.Index] = None
        self.roaring_index = roaring_index

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 0) Empty prefilter (no filter at all, flat index)
    # ------------------------------------------------------------------

    def search_unfiltered_flat(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> list[int]:
        """
        Baseline: no filter, exact ANN over the full flat index.
        """
        assert self.index is not None, "Please call build_index() before searching."
        distances, ids = self.index.search(query_vector, k)
        return [int(pid) for pid in ids[0] if pid != -1]

    # ------------------------------------------------------------------
    # 1) Raw postfilter: Flat ANN → SQL filter
    # ------------------------------------------------------------------

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
        results: list[int] = []
        for pid in candidate_ids:
            if pid in filtered_allowed_set:
                results.append(pid)

        return results

    # ------------------------------------------------------------------
    # 2) IVFPQ postfilter: IVFPQ ANN → SQL filter
    # ------------------------------------------------------------------

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
        results: list[int] = []
        for pid in candidate_ids:
            if pid in filtered_allowed_set:
                results.append(pid)

        return results

    # ------------------------------------------------------------------
    # 3) Roaring bitmap postfilter: Flat ANN → Roaring filter
    # ------------------------------------------------------------------

    def postfilter_search_roaring(
        self,
        query_vector: np.ndarray,
        k: int,
        filter: Dict,
    ) -> list[int]:
        """
        Post-filtering using Roaring bitmaps instead of SQL:
        1) vector search over the full flat index,
        2) intersect the candidate IDs with a Roaring bitmap for the filter.
        """
        assert self.index is not None, "Please call build_index() before searching."
        assert self.roaring_index is not None, "RoaringIndex not configured."

        distances, ids = self.index.search(query_vector, k)
        candidate_ids = ids[0].tolist()

        # Roaring: precomputed set of **allowed** product_ids for this filter
        allowed_ids = self.roaring_index.get_ids_for_filter(filter)  # Set[int]

        results: list[int] = []
        for pid in candidate_ids:
            if pid in allowed_ids:
                results.append(pid)

        return results

    # ------------------------------------------------------------------
    # 4) IVFPQ + Roaring postfilter: IVFPQ ANN → Roaring filter
    # ------------------------------------------------------------------

    def postfilter_search_ivfpq_roaring(
        self,
        query_vector: np.ndarray,
        k: int,
        filter: Dict,
    ) -> list[int]:
        """
        1) Global IVFPQ ANN over all vectors
        2) Post-filter the candidate IDs using Roaring bitmaps
        """
        assert self.ivfpq_index is not None, "Please call build_ivfpq_index() first."
        assert self.roaring_index is not None, "RoaringIndex not configured."

        distances, ids = self.ivfpq_index.search(query_vector, k)
        candidate_ids = ids[0].tolist()

        allowed_ids = self.roaring_index.get_ids_for_filter(filter)  # Set[int]

        return [pid for pid in candidate_ids if pid in allowed_ids]
