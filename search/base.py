from abc import ABC, abstractmethod
from index import Index
from db import DB
from encoder import Encoder
from vector_store import VectorStore
import numpy as np
from tqdm import tqdm
import json
from typing import List, Dict, Any
from .utils import select_filter_by_selectivity, build_filter_from_spec
import time


class Search(ABC):
    def __init__(
        self,
        db: DB,
        encoder: Encoder,
        vector_store: VectorStore,
        method_title: str,
        method_name: str,
        index: Index = None,
    ):
        self.db = db
        self.encoder = encoder
        self.vector_store = vector_store
        self.method_name = method_name
        self.method_title = method_title
        self.index = index

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int, filter: Dict, *args, **kwargs) -> List[int]:
        pass

    def evaluate(
        self,
        qid_pid_filter_list: List[tuple],
        all_query_vectors: np.ndarray,
        selectivity_targets: List[tuple],
        k_fetch: int,
    ) -> List[Dict[str, Any]]:
        """
        Unified evaluation loop for any search method.

        Args:
            qid_pid_filter_list: List of (query_id, product_id, filters_json) tuples. We will evaluate the querying performance for each tuple in this list.
            all_query_vectors: Numpy array of query vectors. Each query_vector is provided to the self.search method.
            selectivity_targets: List of (level_name, target_percent, selectivity_range) tuples.
            k_fetch: Number of results to fetch.

        Returns:
            List of result dictionaries for each selectivity level.
        """
        all_results = []

        for level_name, target_percent, selectivity_range in selectivity_targets:
            latencies_ms = []
            hits = 0
            reciprocal_ranks = []  # For MRR calculation
            gtsims = []
            total_queries = 0
            result_set_size = []

            for i in tqdm(
                range(len(qid_pid_filter_list)),
                desc=f"Evaluating {self.method_name} (Selectivity ~{target_percent}%)",
            ):
                query_id, ground_truth_product_id_str, filters_json_string = (
                    qid_pid_filter_list[i]
                )
                ground_truth_product_id = int(ground_truth_product_id_str)

                query_vector = all_query_vectors[i : i + 1, :]

                try:
                    filter_suite = json.loads(filters_json_string)
                except json.JSONDecodeError:
                    filter_suite = []

                selected_spec = select_filter_by_selectivity(
                    filter_suite, target_percent, selectivity_range
                )
                dynamic_filter = build_filter_from_spec(selected_spec)

                if not dynamic_filter:
                    continue

                total_queries += 1
                start_time = time.perf_counter()

                # Call the search function - returns a ranked list

                final_result_pids = self.search(query_vector=query_vector, k=k_fetch, filter=dynamic_filter)

                end_time = time.perf_counter()
                latencies_ms.append((end_time - start_time) * 1000)
                result_set_size.append(len(final_result_pids))

                # Compute ground truth similarity
                gt_vector = self.vector_store.get_vector_by_product_id(
                    ground_truth_product_id
                )
                gt_sim = (
                    (query_vector @ gt_vector.T)[0, 0]
                    if gt_vector is not None
                    else -1.0
                )
                gtsims.append(gt_sim)

                # Check if hit and compute reciprocal rank
                try:
                    rank = (
                        final_result_pids.index(ground_truth_product_id) + 1
                    )  # 1-indexed
                    reciprocal_ranks.append(1.0 / rank)
                    hits += 1
                except ValueError:
                    # Ground truth not in results
                    reciprocal_ranks.append(0.0)

            if total_queries == 0:
                print(
                    f"Warning: No queries processed for level {level_name}. Check if filters exist."
                )

            recall = (hits / total_queries) if total_queries > 0 else 0
            mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
            avg_result_set_size = np.mean(result_set_size) if result_set_size else 0
            avg_latency = np.mean(latencies_ms) if latencies_ms else 0
            p95_latency = np.percentile(latencies_ms, 95) if latencies_ms else 0
            avg_gt_sim = np.mean(gtsims) if gtsims else 0

            all_results.append(
                {
                    "level": level_name,
                    "target_selectivity": f"~{target_percent}%",
                    "total_queries": total_queries,
                    "hits": hits,
                    "recall": recall,
                    "mrr": mrr,
                    "avg_result_set_size": avg_result_set_size,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "avg_gt_sim": avg_gt_sim,
                }
            )

        return all_results

    def log_results_summary(
        self,
        all_results: List[Dict[str, Any]],
        m_factor: int,
        k_fetch: int,
        out_path: str,
    ) -> None:
        """
        Append formatted summary table for evaluation results to a file.
        """
        lines = []

        lines.append(f"\n--- FINAL SUMMARY: {self.method_title} ---")
        lines.append("-" * 95)
        lines.append(f"M_FACTOR (Overfetch): {m_factor} (K_FETCH={k_fetch})")
        lines.append(
            f"{'Level':<18} | {'Recall':<8} | {'MRR':<8} | {'Avg RSS':<8} | "
            f"{'P95 Lat (ms)':<12} | {'Avg Lat (ms)':<12} | {'Hits':<5}"
        )
        lines.append("-" * 95)

        for metrics in all_results:
            lines.append(
                f"{metrics['level']:<18} | {metrics['recall']:<8.4f} | {metrics['mrr']:<8.4f} | "
                f"{metrics['avg_result_set_size']:<8.1f} | {metrics['p95_latency_ms']:<12.2f} | "
                f"{metrics['avg_latency_ms']:<12.2f} | {metrics['hits']:<5}"
            )

        lines.append("-" * 95)
        lines.append("")

        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))