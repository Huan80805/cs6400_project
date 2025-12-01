import argparse
from typing import List
from index import FlatL2, IVFPQ

# Import our classes
from db import DB
from encoder import Encoder
from vector_store import VectorStore
from query_vector_store import QueryVectorStore
from bitmap.roaring_index import RoaringIndex

from search import (
    Search,
    FlatPostfilter,
    FlatPrefilter,
    FlatPostfilterRoaring,
    FlatPrefilterRoaring,
    IVFPQPostfilter,
    IVFPQPostfilterRoaring,
    IVFPQPrefilter,
    IVFPQPrefilterRoaring,
)


def main():
    parser = argparse.ArgumentParser(description="Run search evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["esci", "amz_c4"],
        default="esci",
        help="Dataset to use for queries: 'esci' or 'amz_c4' (default: esci)",
    )
    args = parser.parse_args()

    DB_PATH = "amz.db"
    EMBEDDINGS_PATH = "embeddings.parquet"

    K_GOAL = 1
    M_FACTOR = 100  # large over-fetch factor for postfiltering
    K_FETCH = K_GOAL * M_FACTOR

    SELECTIVITY_TARGETS = [
        ("Low (<1%)", 0.1, (0.0, 1.0)),
        ("Low-Mid (1-10%)", 1.0, (1.0, 10.0)),
        ("Mid (10-50%)", 10.0, (10.0, 50.0)),
        ("High (>50%)", 50.0, (50.0, 101.0)),  # Use 101 to be inclusive
    ]
    db = DB(path=DB_PATH)
    encoder = Encoder()
    # Use different bitmap file for different datasets
    ROARING_PATH = f"bitmaps_{args.dataset}.pkl"
    roaring = RoaringIndex(ROARING_PATH)
    vector_store = VectorStore(EMBEDDINGS_PATH, db)
    print("Building index...")
    flat_l2_index = FlatL2(
        vector_store.dims, vector_store.vectors, vector_store.product_ids
    )
    print("Built index, now building IVFPQ index...")
    ivfpq_index = IVFPQ(vector_store.vectors, vector_store.product_ids)
    print("IVFPQ index build complete.")

    query_vector_store = QueryVectorStore(args.dataset, encoder, db)
    qid_pid_filter_list, all_query_vectors = query_vector_store.load_query_embeddings()

    search_instances: List[Search] = []
    # ============================================================
    # 1) Raw prefilter: SQL filter → Flat ANN
    # ============================================================
    search_instances.append(
        FlatPrefilter(db, encoder, vector_store, flat_l2_index, rebuild_index=False)
    )

    # ============================================================
    # 2) IVFPQ prefilter: SQL filter → IVFPQ ANN
    # ============================================================
    search_instances.append(
        IVFPQPrefilter(db, encoder, vector_store, ivfpq_index, rebuild_index=False)
    )

    # ============================================================
    # 3) Roaring bitmap prefilter: Roaring filter → Flat ANN
    # ============================================================
    search_instances.append(
        FlatPrefilterRoaring(
            db, encoder, vector_store, roaring, flat_l2_index, rebuild_index=False
        )
    )

    # ============================================================
    # 4) IVFPQ + Roaring bitmap prefilter: Roaring filter → IVFPQ ANN
    # ============================================================
    search_instances.append(
        IVFPQPrefilterRoaring(
            db, encoder, vector_store, roaring, ivfpq_index, rebuild_index=False
        )
    )

    # ============================================================
    # 5) Raw postfilter: Flat ANN → SQL filter
    # ============================================================
    search_instances.append(FlatPostfilter(db, encoder, vector_store, flat_l2_index))

    # ============================================================
    # 6) IVFPQ postfilter: IVFPQ ANN → SQL filter
    # ============================================================
    search_instances.append(IVFPQPostfilter(db, encoder, vector_store, ivfpq_index))

    # ============================================================
    # 7) Roaring bitmap postfilter: Flat ANN → Roaring filter
    # ============================================================
    search_instances.append(
        FlatPostfilterRoaring(db, encoder, vector_store, flat_l2_index, roaring)
    )

    # ============================================================
    # 8) IVFPQ + Roaring bitmap postfilter: IVFPQ → ANN Roaring filter
    # ============================================================
    search_instances.append(
        IVFPQPostfilterRoaring(db, encoder, vector_store, ivfpq_index, roaring)
    )

    for s in search_instances:
        result = s.evaluate(
            qid_pid_filter_list, all_query_vectors, SELECTIVITY_TARGETS, K_FETCH
        )
        s.log_results_summary(result, M_FACTOR, K_FETCH, "results.txt")

    db.close()


if __name__ == "__main__":
    main()
