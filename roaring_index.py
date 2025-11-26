from typing import Dict, Tuple, Any, Set
from pyroaring import BitMap
import pickle
from bitmap_keys import make_key  # NEW

class RoaringIndex:
    """
    Holds Roaring BitMaps keyed by a (column, op, value) signature.

    You build this OFFLINE from the filters JSON and then load it at runtime.
    """
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self._bitmaps: dict[str, BitMap] = pickle.load(f)
        print(f"RoaringIndex loaded: {len(self._bitmaps)} keys")

    def get_ids_for_filter(self, filter_dict: Dict[str, Tuple[str, Any]]) -> Set[int]:
        """
        Given a dynamic_filter like {"average_rating": ("BETWEEN", (3.5, 3.7))},
        return the set of product_ids that satisfy it, using Roaring bitmaps.
        """
        if not filter_dict:
            return set()

        col, (op, val) = next(iter(filter_dict.items()))
        key = make_key(col, op, val)

        bm: BitMap | None = self._bitmaps.get(key)
        if bm is None:
            return set()
        return set(bm)
