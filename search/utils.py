from typing import Any, Optional, Set, Tuple, List, Dict
import numpy as np


def select_filter_by_selectivity(
    filter_list: list[dict[str, Any]],
    target_percent: float,
    selectivity_range: tuple[float, float],
) -> Optional[dict[str, Any]]:
    """
    Selects one filter from the list that falls within the valid
    selectivity_range AND is closest to the target_percent.
    """
    if not filter_list:
        return None

    min_perc, max_perc = selectivity_range

    # 1. Find all valid filters *within the correct bucket*
    valid_filters = []
    for f in filter_list:
        perc = f.get("match_percentage")
        if perc is not None and min_perc <= perc < max_perc:
            valid_filters.append(f)

    # 2. If no filters are in this bucket, we can't use one.
    if not valid_filters:
        return None

    # 3. From the valid subset, find the one closest to our ideal target.
    selected_filter = min(
        valid_filters,
        key=lambda f: abs(f.get("match_percentage", 101) - target_percent),
    )
    return selected_filter


def build_filter_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Takes: {"filter_column": "average_rating", "filter_value": [3.5, 3.7], ...}
    Returns: {"average_rating": ("BETWEEN", (3.5, 3.7))}
    """
    if not spec:
        return {}

    col = spec.get("filter_column")
    val = spec.get("filter_value")

    if col is None or val is None:
        return {}

    # Handle range/list values as BETWEEN
    if isinstance(val, list) and len(val) == 2:
        return {col: ("BETWEEN", tuple(val))}

    # Handle JSON/text LIKE searches
    if col in ("features_json", "details_json"):
        return {col: ("LIKE", f"%{val}%")}

    # Default to simple equality
    return {col: ("=", val)}
