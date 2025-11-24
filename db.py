import sqlite3
import sys
from typing import List, Tuple, Dict, Set, Optional
import numpy as np


class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def count_products(self, categories: Optional[List[str]]) -> int:
        cur = self.conn.cursor()
        if categories:
            q = f"SELECT COUNT(*) FROM products WHERE main_category IN ({','.join(['?'] * len(categories))})"
            cur.execute(q, categories)
        else:
            cur.execute("SELECT COUNT(*) FROM products")
        return int(cur.fetchone()[0])

    def fetch_products_page(
        self, start: int, limit: int, categories: Optional[List[str]]
    ) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        if categories:
            q = (
                "SELECT product_id, parent_asin, main_category, product_title AS title, features_json AS features_text, details_json AS details_text\n"
                "FROM products\n"
                f"WHERE main_category IN ({','.join(['?'] * len(categories))})\n"
                "ORDER BY product_id ASC\n"
                "LIMIT ? OFFSET ?"
            )
            cur.execute(q, (*categories, limit, start))
        else:
            q = (
                "SELECT product_id, parent_asin, main_category, product_title AS title, features_json AS features_text, details_json AS details_text\n"
                "FROM products\n"
                "ORDER BY product_id ASC\n"
                "LIMIT ? OFFSET ?"
            )
            cur.execute(q, (limit, start))
        rows = cur.fetchall()
        return rows

    def fetch_topk_reviews(self, parent_asin: str, k: int) -> List[Tuple[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(review_title, '') AS title, COALESCE(review_text, '') AS text
            FROM reviews
            WHERE parent_asin = ?
            ORDER BY helpful_vote DESC
            LIMIT ?
            """,
            (parent_asin, k),
        )
        return [(r[0], r[1]) for r in cur.fetchall()]

    def load_esci_queries(
        self, categories: Optional[List[str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Use 'E' label
        Returns list of (query_id, query_text, ground_truth_product_id, filters_json)
        ground_truth_product_id is parent_asin
        """
        cur = self.conn.cursor()
        if categories:
            cur.execute(
                f"""
                SELECT eq.query_id, eq.query, p.product_id, p.filters_json
                FROM esci_queries eq
                JOIN products p 
                ON eq.product_id  = p.parent_asin
                WHERE esci_label = 'E' AND small_version = 1 AND p.main_category IN ({",".join(["?"] * len(categories))})
                """,
                categories,
            )
        else:
            cur.execute(
                """
                SELECT eq.query_id, eq.query, p.product_id, p.filters_json
                FROM esci_queries eq
                JOIN products p 
                ON eq.product_id  = p.parent_asin
                WHERE esci_label = 'E' AND small_version = 1
                """
            )

        queries = cur.fetchall()
        return [
            (r["query_id"], r["query"], r["product_id"], r["filters_json"])
            for r in queries
        ]

    def _build_filter_clause(self, filter_dict: Dict) -> Tuple[str, tuple]:
        """
        Helper to build SQL WHERE clauses and params from a filter dict.
        Example: {"average_rating": ("BETWEEN", (3.5, 3.7))}
        Returns: ("AND average_rating BETWEEN ? AND ?", (3.5, 3.7))
        """
        clauses = []
        params = []
        for col, op_val in filter_dict.items():
            op, val = op_val

            # Basic validation to prevent SQL injection
            if not all(c.isalnum() or c == "_" for c in col):
                raise ValueError(f"Invalid column name: {col}")

            # --- UPDATED: Added 'BETWEEN' ---
            if op.upper() not in (
                ">=",
                "<=",
                "=",
                ">",
                "<",
                "!=",
                "IN",
                "LIKE",
                "BETWEEN",
            ):
                raise ValueError(f"Invalid operator: {op}")

            if op.upper() == "BETWEEN":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError(
                        f"BETWEEN operator requires a 2-element list/tuple. Got {val}"
                    )
                clauses.append(f"AND {col} BETWEEN ? AND ?")
                params.extend(val)  # Add both values to params
            else:
                clauses.append(f"AND {col} {op} ?")
                params.append(val)

        return " ".join(clauses), tuple(params)

    def get_filtered_ids(self, candidate_ids: List[int], filter: Dict) -> Set[int]:
        """
        Filters a list of candidate_ids against the DB.
        Used for post-filtering.
        """
        if not candidate_ids:
            return set()

        cur = self.conn.cursor()
        placeholders = ",".join(["?"] * len(candidate_ids))

        # Build the dynamic filter clause
        where_clause, filter_params = self._build_filter_clause(filter)

        sql = f"""
           SELECT product_id FROM products
           WHERE product_id IN ({placeholders})
           {where_clause}
        """

        # Combine candidate IDs and filter params
        params = (*candidate_ids, *filter_params)

        cur.execute(sql, params)
        final_ids = {row[0] for row in cur.fetchall()}
        return final_ids

    def close(self):
        self.conn.close()
